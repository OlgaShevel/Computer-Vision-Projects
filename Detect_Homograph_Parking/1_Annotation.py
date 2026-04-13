
# Делать в colab (там проще создать интеракт среду, чем на локалке)

from google.colab import drive
drive.mount('/content/drive')


# 1-я ячейка - создание интеракт среды
!pip install ipympl -q

from google.colab import output

# Включить встроенную поддержку сторонних виджетов
output.enable_custom_widget_manager()
%matplotlib widget


# 2-я ячейка

# тут все верно - этот =========================================================

# код для формирования геоджейсона
# ЭТОТ ВЕРНЫЙ БЕЗ ДУБЛИРВАНИЯ ТОЧКИ

from PIL import Image
import matplotlib.pyplot as plt
import json
import os
from google.colab import files

def annotate_parking_polygons(
    image_path,
    save_dir,
    max_spaces=100
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    img = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(img)
    ax.set_title('Разметка мышкой')
    ax.grid(alpha=0.3)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    geojson_filename = os.path.join(save_dir, base_name + '.geojson')

    spaces = []
    current_points = []
    current_id = 1
    finished = [False]

    poly_lines = []
    point_scatters = []
    text_labels = []
    status_text = [None]

    def update_status():
        if status_text[0]:
            status_text[0].remove()
        txt = ax.text(0.02, 0.98, f'Готово мест: {len(spaces)} | Следующий ID: {current_id}',
                       transform=ax.transAxes, fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        status_text[0] = txt

    def save_to_geojson():
        features = []
        for space in spaces:
            contour = space['contour'].copy()
            if contour and contour[-1] != contour[0]:  # ✅ ИСПРАВЛЕНО: проверка перед добавлением
                contour.append(contour[0])
            features.append({
                "type": "Feature",
                "properties": {"id": space['id']},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [contour]
                }
            })

        geojson = {"type": "FeatureCollection", "features": features}

        with open(geojson_filename, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)
        print(f"\nСохранено в Drive: {geojson_filename}")

        download_path = f"/content/{base_name}.geojson"
        with open(download_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)
        files.download(download_path)
        print(f"Файл скачан: {base_name}.geojson")
        print(f"Всего мест в файле: {len(features)}")

    def redraw():
        for line in poly_lines: line.remove()
        for scat in point_scatters: scat.remove()
        for txt in text_labels: txt.remove()
        poly_lines.clear()
        point_scatters.clear()
        text_labels.clear()

        for space in spaces:
            contour = space['contour']
            contour_closed = contour + [contour[0]]
            xs, ys = zip(*contour_closed)
            line = ax.plot(xs, ys, 'lime', linewidth=2)[0]
            poly_lines.append(line)
            xs, ys = zip(*contour)
            scat = ax.scatter(xs, ys, c='lime', s=10, zorder=5)
            point_scatters.append(scat)

            offset_x = 13
            offset_y = -5

            txt = ax.text(contour[0][0] + offset_x, contour[0][1] + offset_y,
                          str(space['id']),
                          color='orange', fontsize=10, zorder=10)

        if current_points:
            xs, ys = zip(*current_points)
            scat = ax.scatter(xs, ys, c='red', s=10, zorder=5)
            point_scatters.append(scat)
            if len(current_points) > 1:
                xs, ys = zip(*current_points)
                line = ax.plot(xs, ys, 'r--', linewidth=1)[0]
                poly_lines.append(line)

        update_status()
        fig.canvas.draw_idle()

    def on_key_press(event):
        nonlocal current_id
        print(f"Нажата клавиша: '{event.key}'")

        if event.key in ['enter', 'return', 'd', 'f', 'D', 'F']:
            if len(current_points) >= 3:
                spaces.append({'id': current_id, 'contour': current_points.copy()})
                print(f"Место #{current_id} завершено ({len(current_points)} точек)")
                current_points.clear()
                current_id += 1
                redraw()
            else:
                print(f"Нужно минимум 3 точки, сейчас {len(current_points)}")

        elif event.key.lower() == 's':
            if current_points and len(current_points) >= 3:
                spaces.append({'id': current_id, 'contour': current_points.copy()})
                print(f"Место #{current_id} добавлено перед сохранением")
            print(f"\nПуть: {geojson_filename}")
            print(f"Всего мест: {len(spaces)}")
            save_to_geojson()
            finished[0] = True
            plt.close()

        elif event.key.lower() == 'q':
            print("Выход без сохранения")
            finished[0] = True
            plt.close()

        elif event.key.lower() == 'c':
            current_points.clear()
            print("↶ Точки отмены")
            redraw()

    def on_click(event):
        if finished[0]:
            return
        if event.inaxes != ax:
            return
        if len(spaces) >= max_spaces:
            print(f"Достигнут максимум мест: {max_spaces}")
            return
        x, y = int(event.xdata), int(event.ydata)
        current_points.append([x, y])
        redraw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    update_status()
    plt.show()

# ==============================================================================

annotate_parking_polygons(
    image_path='/content/drive/MyDrive/PARKING/POINT_100/2012-09-12_06_05_16.jpg', # Заменить на свой путь
    save_dir='/content/drive/MyDrive/PARKING/POINT_100/Annotations',               # Заменить на свой путь
    max_spaces=100
)

# D или F - завершить место
# C  - отмена
# S - сохранить
# Q - выйти