import cv2
import numpy as np
import json
import os
from pathlib import Path

# Автопути относительно расположения файлов
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "test-task"
SPLIT_FILE = DATA_ROOT / "split.json"
OUTPUT_DIR = SCRIPT_DIR

with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    split_data = json.load(f)

train_sessions = split_data.get("train", [])
val_sessions = split_data.get("val", [])

ref_src = {"top": [], "bottom": []}
ref_dst = {"top": [], "bottom": []}

print("Сбор ручных точек из train...")
for sess_rel in train_sessions:
    sess_dir = DATA_ROOT / sess_rel
    for cam in ["top", "bottom"]:
        coords_path = sess_dir / f"coords_{cam}.json"
        if not coords_path.exists(): continue
        with open(coords_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for pair in data:
            if cam in pair["file1_path"]:
                src_list = pair["image1_coordinates"]
                dst_list = pair["image2_coordinates"]
            else:
                src_list = pair["image2_coordinates"]
                dst_list = pair["image1_coordinates"]
            for s, d in zip(src_list, dst_list):
                ref_src[cam].append([s["x"], s["y"]])
                ref_dst[cam].append([d["x"], d["y"]])

for cam in ["top", "bottom"]:
    src_arr = np.array(ref_src[cam], dtype=np.float32)
    dst_arr = np.array(ref_dst[cam], dtype=np.float32)
    np.savez(OUTPUT_DIR / f"{cam}_ref.npz", src=src_arr, dst=dst_arr)
    print(f"Сохранено {cam}_ref.npz: {len(src_arr)} пар")

# Функция predict 
def predict(x, y, source, K=5):
    src_pts = np.array(ref_src[source], dtype=np.float32)
    dst_pts = np.array(ref_dst[source], dtype=np.float32)
    if len(src_pts) == 0: return float(x), float(y)
    query = np.array([x, y], dtype=np.float32)
    dists = np.linalg.norm(src_pts - query, axis=1)
    idx = np.argsort(dists)[:K]
    p_src, p_dst = src_pts[idx], dst_pts[idx]
    try:
        H, status = cv2.findHomography(p_src, p_dst, method=cv2.RANSAC, ransacReprojThreshold=1.0)
        if H is None or status is None: raise ValueError
        mask = status.ravel().astype(bool)
        if mask.sum() < 4: raise ValueError
        H, _ = cv2.findHomography(p_src[mask], p_dst[mask], cv2.RANSAC)
        mapped = cv2.perspectiveTransform(p_src[mask].reshape(-1, 1, 2), H)[:, 0]
        dists_local = np.linalg.norm(p_src[mask] - query, axis=1) + 1e-6
        weights = 1.0 / dists_local; weights /= weights.sum()
        bias = np.sum(weights[:, None] * (p_dst[mask] - mapped), axis=0)
        pred = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), H)[0][0]
        return float(pred[0] + bias[0]), float(pred[1] + bias[1])
    except Exception:
        fb = query + np.mean(p_dst - p_src, axis=0)
        return float(fb[0]), float(fb[1])

# Валидация на val сплите
print("\nРасчёт метрики на val...")
errors_top, errors_bottom = [], []
for sess_rel in val_sessions:
    sess_dir = DATA_ROOT / sess_rel
    for cam in ["top", "bottom"]:
        coords_path = sess_dir / f"coords_{cam}.json"
        if not coords_path.exists(): continue
        with open(coords_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for pair in data:
            if cam in pair["file1_path"]:
                src_list = pair["image1_coordinates"]
                dst_list = pair["image2_coordinates"]
            else:
                src_list = pair["image2_coordinates"]
                dst_list = pair["image1_coordinates"]
            for s, d in zip(src_list, dst_list):
                px, py = predict(s["x"], s["y"], cam)
                err = np.hypot(px - d["x"], py - d["y"])
                if cam == "top": errors_top.append(err)
                else: errors_bottom.append(err)

med_top = float(np.mean(errors_top)) if errors_top else 0.0
med_bottom = float(np.mean(errors_bottom)) if errors_bottom else 0.0
metrics = {"top_to_door2_med": med_top, "bottom_to_door2_med": med_bottom, "overall_med": (med_top+med_bottom)/2}
with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(f"MED top -> door2:    {med_top:.2f} px")
print(f"MED bottom -> door2: {med_bottom:.2f} px")
print("Готово. Артефакты и metrics.json сохранены в solution/")