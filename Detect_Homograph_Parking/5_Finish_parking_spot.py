
#pip install onnxruntime-gpu


import os, json, numpy as np, cv2, onnxruntime as ort
from matplotlib.path import Path
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_parking_100spots.onnx')


def check_parking_spot(geojson_path, calibrate_json_path, image_path, output_dir=None):
    """
    Проверяет, занято ли парковочное место.
    Вход:
        1. geojson_path — файл разметки места (например, park_88.geojson)
        2. calibrate_json_path — файл калибровки камеры (calibrate_camera_01.json)
        3. image_path — тестовое изображение
        4. output_dir — куда сохранять JSON
    """

    # Извлечь park_idx из имени файла geojson
    # park_88.geojson - 88
    geojson_filename = os.path.basename(geojson_path)
    park_idx = int(''.join(filter(str.isdigit, geojson_filename)))

    # Извлечь camera_idx из имени файла калибровки
    # calibrate_camera_01.json - 1
    calibrate_filename = os.path.basename(calibrate_json_path)
    calibrate_idx = int(''.join(filter(str.isdigit, calibrate_filename)))

    # Загрузить 4 точки из калибровки и посчитать матрицу H
    with open(calibrate_json_path, 'r') as f:
        calibrate_data = json.load(f)

    points = calibrate_data.get('points', [])

    pixel_pts = []
    world_pts = []
    for p in points:
        pixel_pts.append([p[0], p[1]])
        world_pts.append([p[2], p[3]])

    pixel_pts = np.array(pixel_pts, dtype=np.float32).reshape(-1, 1, 2)
    world_pts = np.array(world_pts, dtype=np.float32).reshape(-1, 1, 2)

    # Посчитать матрицу H из 4 точек
    H, _ = cv2.findHomography(pixel_pts, world_pts)

    # Загрузить разметку машиноместа
    with open(geojson_path, 'r') as f:
        spot_data = json.load(f)
    polygon_coords = spot_data['features'][0]['geometry']['coordinates'][0]
    spot_polygon = Path(polygon_coords)

    # Модель
    model_path = MODEL_PATH
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # Загрузить изображение
    img = cv2.imread(image_path)
    orig_h, orig_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Инференс
    IMG_SIZE = 640
    scale = min(IMG_SIZE / orig_w, IMG_SIZE / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    dx = (IMG_SIZE - new_w) // 2
    dy = (IMG_SIZE - new_h) // 2
    canvas[dy:dy+new_h, dx:dx+new_w] = resized
    input_tensor = canvas.astype(np.float32) / 255.0
    input_tensor = input_tensor[:, :, ::-1].transpose(2, 0, 1)[np.newaxis, ...]
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})[0]

    # Проверка машиноместа
    occupied = False
    occupied_bbox = None

    # Полигон машиноместа
    spot_poly_pixel = Path(polygon_coords)

    for i in range(outputs.shape[2]):
        conf = outputs[0, 4, i]
        if conf >= 0.55:
            xc, yc, w, h = outputs[0, :4, i]
            x_min = ((xc - w/2) - dx) / scale
            y_min = ((yc - h/2) - dy) / scale
            x_max = ((xc + w/2) - dx) / scale
            y_max = ((yc + h/2) - dy) / scale
            bbox = [x_min, y_min, x_max, y_max]

            # Центр бокса в пикселях
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            # Проверяем, попадает ли центр в полигон места (в пикселях)
            if spot_poly_pixel.contains_point((center_x, center_y)):
                occupied = True
                occupied_bbox = bbox
                break


    # Визуализация
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(img_rgb)
    ax.axis('off')

    # Рисуем красный ббокс, если место занято
    if occupied_bbox:
        x_min, y_min, x_max, y_max = map(int, occupied_bbox)
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                              linewidth=2, edgecolor='red', facecolor='none', linestyle='-')
        ax.add_patch(rect)

    # Рисуем полигон места (всегда, независимо от статуса)
    spot_color = 'red' if occupied else 'lime'
    spot_poly = plt.Polygon(polygon_coords, closed=True, linewidth=4,
                            edgecolor=spot_color, facecolor=spot_color, alpha=0.3)
    ax.add_patch(spot_poly)


    # Подпись с номером места (всегда!)
    # Находим верхнюю точку полигона, чтобы разместить номер над ним
    poly_y_coords = [p[1] for p in polygon_coords]
    label_y = min(poly_y_coords)  # самая верхняя точка

    # Размещаем номер чуть выше полигона
    ax.text(
        np.mean([p[0] for p in polygon_coords]),   # X: центр полигона по горизонтали
        label_y - 30,                              # Y: на 10 пикселей выше верхней точки
        f'{park_idx}',                             # Текст: номер места
        color='black',                             # Цвет текста (универсальный)
        fontsize=10,
        weight='bold',
        ha='center',                               # Выравнивание по центру
        bbox=dict(
            boxstyle='round, pad=0.3',
            facecolor='yellow',
            edgecolor='gray',
            alpha=0.9
        )
    )

    # КОНЕЦ НОВОГО БЛОКА
    status = f'- ЗАНЯТО (true)' if occupied else '- СВОБОДНО (false)'
    ax.set_title(f'\nМесто: № {park_idx} {status}', fontsize=12, pad=12)
    plt.tight_layout()
    plt.show()

# Формирование json
    # # Извлечь calibrate_idx и test_idx из имени файла
    # # Пример: "test_calibrate_camera_1_test_15.jpg"

    img_filename = os.path.basename(image_path)

    try:
        parts = img_filename.replace('.jpg', '').split('_')
        calibrate_idx = int(parts[3])
        test_idx = int(parts[-1])

    except:
        test_idx = 1
        calibrate_idx = 1

    result = {
        "params": {
            "park_idx": park_idx,
            "calibrate_idx": calibrate_idx
        },
        "result": {
            str(test_idx): {
                "detected": occupied
            }
        }
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'result_{park_idx}_{test_idx}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        print(f"JSON сохранён: {output_path}")
        return output_path 
    


# Вызов функции, вывод инфо для нескольких паркомест на одном фото 

if __name__ == "__main__":
    try:
        # Вызов функции, вывод инфо для нескольких паркомест на одном фото 
        BASE = BASE_DIR

        # Ввод данных
        numbers_park = list(map(int, input("Номера машиномест: ").split()))
        number_files = int(input("Номер файла test_calibrate_camera_1_test_.jpg: "))

        # Количество файлов
        files = next(os.walk(r'C:\Users\ForPython\PROJECTS\Detect_Homograph_Parking\test'))[2]
        max_file_num = len(files)

        # Проверка диапазона для ловли ошибок
        if number_files > max_file_num:
            print(f"\033[1m\033[91mОшибка: введите номер не более {max_file_num}\033[0m")
        else:
            for park_num in numbers_park:
                result = check_parking_spot(
                    geojson_path=f'{BASE}/ALL_100_geojson/park_{park_num}.geojson',
                    calibrate_json_path=f'{BASE}/calibrate_camera_1.json',
                    image_path=f'{BASE}/test/test_calibrate_camera_1_test_{number_files}.jpg',
                    output_dir=f'{BASE}/Test_Results'
                )
    except ValueError:
        print("\033[1m\033[91mОшибка: введите целое число\033[0m")
    except Exception as e:
        print(f"\033[1m\033[91mНепредвиденная ошибка: {e}\033[0m")