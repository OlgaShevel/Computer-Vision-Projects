# Калибровка фронтальной камеры, вид на парковку на 100 мест

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

# GeoJSON с разметкой машиномест на парковке в мировых координатах. 
# Укажите свой путь к файлу
with open(r'C:\Users\ForPython\PROJECTS\Detect_Homograph_Parking\ROI_Points4_2012-09-12_06_05_16.geojson') as f:
    geo = json.load(f)

# 4 точки калибровки, файл c 4 точками по углам пространства праковки в пиксельных координатах.
# Укажите свой путь к файлу
with open(r'C:\Users\ForPython\PROJECTS\Detect_Homograph_Parking\calibrate_naoborot.json') as f:
    calib = json.load(f)
camera_points = np.array(calib['camera_points'], dtype=np.float32)

# 4 точки соответствия с json из GeoJSON
geo_points = np.array(geo['features'][0]['geometry']['coordinates'][0][:4], dtype=np.float32)

# Гомография
H, mask = cv2.findHomography(camera_points, geo_points, cv2.RANSAC)

# Сохранить файл с H
np.save(r'C:\Users\ForPython\PROJECTS\Detect_Homograph_Parking\homography_matrix.npy', H)
h_dict = {"H": H.tolist(), "mask": mask.ravel().tolist()}
with open(r'C:\Users\ForPython\PROJECTS\Detect_Homograph_Parking\Calibrate_camera.json', 'w') as f:
    json.dump(h_dict, f, indent=2)

print("СОХРАНЕНО:")
print("homography_matrix.npy")
print("Calibrate_camera.json")

# Тест на фото
img = cv2.imread(r'C:\Users\ForPython\PROJECTS\Detect_Homograph_Parking\2012-09-12_06_05_16.jpg')


height, width = img.shape[:2]
top_view = cv2.warpPerspective(img, H, (width, height))

# Наложить разметку парковок на фото оригинала
img_with_parking = img.copy()
h_inv = np.linalg.pinv(H)

# Изображение разметки
for i, feature in enumerate(geo['features'][1:]):
    points = np.array(feature['geometry']['coordinates'][0], dtype=np.float32)
    points_img = cv2.perspectiveTransform(points.reshape(-1,1,2), h_inv).reshape(-1,2).astype(np.int32)
    cv2.polylines(img_with_parking, [points_img], True, (0, 255, 0), 3)
    cv2.putText(img_with_parking, str(i+1), points_img[0].astype(np.int32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Визуализация
plt.figure(figsize=(20, 6))

plt.subplot(131)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('ОРИГИНАЛ')
plt.axis('off')

plt.subplot(132)
plt.imshow(cv2.cvtColor(top_view, cv2.COLOR_BGR2RGB))
plt.title(f'ПЛАН\nМаска: {mask.ravel().tolist()}')
plt.axis('off')

plt.tight_layout()
plt.show()

# Проверка точности
test_points = camera_points.reshape(-1,1,2)
transformed = cv2.perspectiveTransform(test_points, H)

print("Сравнение точек для Плана:")
for i in range(4):
    print(f"{camera_points[i]} → {transformed[i][0]}")