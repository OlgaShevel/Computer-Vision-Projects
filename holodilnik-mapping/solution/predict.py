import numpy as np
import cv2
import os
from pathlib import Path

# Автопоиск артефактов в той же папке, где лежит файл
MODEL_DIR = Path(__file__).resolve().parent

_MODELS = {}
for cam in ["top", "bottom"]:
    path = MODEL_DIR / f"{cam}_ref.npz"
    if path.exists():
        data = np.load(path)
        _MODELS[cam] = {"src": data["src"], "dst": data["dst"]}

def predict(x, y, source, K=5):
    if source not in _MODELS:
        raise ValueError(f"Не найден справочник для '{source}'. Запустите train.py.")
    
    src_pts = _MODELS[source]["src"]
    dst_pts = _MODELS[source]["dst"]
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
        weights = 1.0 / dists_local
        weights /= weights.sum()
        bias = np.sum(weights[:, None] * (p_dst[mask] - mapped), axis=0)
        
        pred = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), H)[0][0]
        return float(pred[0] + bias[0]), float(pred[1] + bias[1])
    except Exception:
        fb = query + np.mean(p_dst - p_src, axis=0)
        return float(fb[0]), float(fb[1])

if __name__ == "__main__":
    print("Тест интерфейса:")
    print("top:", predict(1000.0, 800.0, "top"))
    print("bottom:", predict(1200.0, 900.0, "bottom"))