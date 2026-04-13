

# Запуск формата onnx на тестовой выборке датасета 

import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import json
from matplotlib.path import Path
import cv2
import onnxruntime as ort

# Замените пути на свои
# test - тестовые фото
# labs_test - лейблы для текстовых фото 
# .onnx - модель в onnx
# calibrate_camera_01.json - файл c 4 точками по углам пространства праковки в пиксельных координатах

TEST_IMAGES = r'C:\Users\ForPython\PROJECTS\Detect_Homograph_Parking\test'
TEST_LABELS = r'C:\Users\ForPython\PROJECTS\Detect_Homograph_Parking\labs_test'
MODEL_ONNX = r'C:\Users\ForPython\PROJECTS\Detect_Homograph_Parking\best_parking_100spots.onnx'
ROI_JSON_PATH = r'C:\Users\ForPython\PROJECTS\Detect_Homograph_Parking\calibrate_camera_01.json'


NUM_IMAGES = 10
CONF_THRESHOLD = 0.55
MIN_BOX_AREA = 0.0005
MAX_BOX_AREA = 0.05


def load_roi_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'roi_polygon' in data:
        coords = data['roi_polygon']
    elif 'camera_points' in data:
        coords = data['camera_points']
    else:
        coords = data

    polygon = [(float(p[0]), float(p[1])) for p in coords]
    return polygon


ROI_POLYGON = load_roi_from_json(ROI_JSON_PATH)
print(f"ROI загружен: {len(ROI_POLYGON)} точек")
print(f"Координаты: {ROI_POLYGON}")


# Определить цвета 
GT_COLOR = (0, 1, 0, 0.8)
PRED_COLOR = (1, 0, 0, 0.8)
ROI_COLOR = (0, 0, 1, 0.3)

# Загрузить модель в onnx
session = ort.InferenceSession(MODEL_ONNX, providers=['CPUExecutionProvider'])
print(f"Модель загружена (ONNX)")

# Прочитать разметку 
def parse_annotation(txt_path, img_width, img_height):
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 5:
                try:
                    class_id = int(float(parts[0]))
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    x_min = (x_center - width / 2) * img_width
                    y_min = (y_center - height / 2) * img_height
                    box_width = width * img_width
                    box_height = height * img_height
                    boxes.append((class_id, x_min, y_min, box_width, box_height))
                except:
                    continue
    return boxes

# Проверить попадание в полигон
def is_in_polygon(x_center, y_center, polygon):
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y_center > min(p1y, p2y):
            if y_center <= max(p1y, p2y):
                if x_center <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y_center - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x_center <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def is_in_roi(x_min, y_min, box_width, box_height, img_width, img_height, polygon):
    x_center = x_min + box_width / 2
    y_center = y_min + box_height / 2
    return is_in_polygon(x_center, y_center, polygon)

# Проверить размер бокса
def is_valid_box_size(box_width, box_height, img_width, img_height, min_area, max_area):
    box_area = (box_width * box_height) / (img_width * img_height)
    return min_area <= box_area <= max_area

# Отрисовать боксы
def draw_boxes(ax, boxes, color, show_conf=False):
    for box in boxes:
        if len(box) == 6:
            class_id, x_min, y_min, box_width, box_height, conf = box
        else:
            class_id, x_min, y_min, box_width, box_height = box
            conf = None
        rect = patches.Rectangle((x_min, y_min), box_width, box_height,
                                  linewidth=2, edgecolor=color[:3], facecolor='none', linestyle='-')
        ax.add_patch(rect)
        if show_conf and conf is not None:
            ax.text(x_min, y_min - 5, f'{conf:.2f}', color=color[:3], fontsize=8, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Отрисоватьт зону ROI 
def draw_roi(ax, roi, img_width, img_height):
    poly = patches.Polygon(roi, closed=True, linewidth=2, edgecolor='blue',
                            facecolor=ROI_COLOR, linestyle='--')
    ax.add_patch(poly)


image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
test_images = [f for f in os.listdir(TEST_IMAGES)
               if os.path.splitext(f)[1].lower() in image_extensions]

if len(test_images) == 0:
    print("Тестовые изображения не найдены")
else:
    selected_images = test_images[:NUM_IMAGES]  # Первые N

    print(f"Найдено тестовых изображений: {len(test_images)}")
    print(f"Показываю {len(selected_images)} изображений\n")

    n_images = len(selected_images)
    n_cols = 2
    n_rows = n_images

    for i, img_name in enumerate(selected_images):
        img_path = os.path.join(TEST_IMAGES, img_name)
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        gt_path = os.path.join(TEST_LABELS, txt_name)

        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        img_width, img_height = img.size

        # === Инференс (тот же код, что был) ===
        img_cv = cv2.imread(img_path)
        orig_h, orig_w = img_cv.shape[:2]
        IMG_SIZE = 640
        scale = min(IMG_SIZE / orig_w, IMG_SIZE / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized = cv2.resize(img_cv, (new_w, new_h))
        canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        dx = (IMG_SIZE - new_w) // 2
        dy = (IMG_SIZE - new_h) // 2
        canvas[dy:dy+new_h, dx:dx+new_w] = resized
        input_tensor = canvas.astype(np.float32) / 255.0
        input_tensor = input_tensor[:, :, ::-1].transpose(2, 0, 1)[np.newaxis, ...]
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})[0]

        pred_boxes = []
        filtered_count = 0
        for j in range(outputs.shape[2]):
            conf = outputs[0, 4, j]
            if conf >= CONF_THRESHOLD:
                xc, yc, w, h = outputs[0, :4, j]
                x_min = ((xc - w/2) - dx) / scale
                y_min = ((yc - h/2) - dy) / scale
                x_max = ((xc + w/2) - dx) / scale
                y_max = ((yc + h/2) - dy) / scale
                box_width = x_max - x_min
                box_height = y_max - y_min

                if not is_valid_box_size(box_width, box_height, img_width, img_height, MIN_BOX_AREA, MAX_BOX_AREA):
                    filtered_count += 1
                    continue
                if not is_in_roi(x_min, y_min, box_width, box_height, img_width, img_height, ROI_POLYGON):
                    filtered_count += 1
                    continue
                pred_boxes.append((0, x_min, y_min, box_width, box_height, conf))

    # === Отрисовка ОДНОГО изображения ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # GT
    ax1.imshow(img_array)
    ax1.axis('off')
    gt_boxes = parse_annotation(gt_path, img_width, img_height)
    draw_boxes(ax1, gt_boxes, GT_COLOR, show_conf=False)
    draw_roi(ax1, ROI_POLYGON, img_width, img_height)
    ax1.set_title(f'{img_name}\nGT: {len(gt_boxes)}', fontsize=10, pad=10)

    # Prediction
    ax2.imshow(img_array)
    ax2.axis('off')
    draw_boxes(ax2, pred_boxes, PRED_COLOR, show_conf=True)
    draw_roi(ax2, ROI_POLYGON, img_width, img_height)
    ax2.set_title(f'Pred: {len(pred_boxes)} (отфильтровано: {filtered_count})', fontsize=10, pad=10)

    plt.tight_layout()
    plt.show()
    print(f"Обработано: {img_name}")