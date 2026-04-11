
from google.colab import drive
drive.mount('/content/drive')



# После того, как ячейка отработает, перезапустить сеанс (в меню) и еще раз запуск ячейки
!pip install ipympl -q

from google.colab import output

# Встроенная поддержка сторонних виджетов
output.enable_custom_widget_manager()
%matplotlib 




# стиль как в CVAT - ЛКМ точка - разверстка кубоида - растащить точки по местам

# Растянуть окно рисовалки
from google.colab import output
output.no_vertical_scroll()


from PIL import Image
import matplotlib.pyplot as plt
import json
import os
from google.colab import files

def annotate_cuboids_cvat_style(
    image_path,
    save_dir,
    max_spaces=100
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    img = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(img)
    ax.set_title('Создание кубоида (псевдо-3D) без CVAT')
    ax.grid(alpha=0.3)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    geojson_filename = os.path.join(save_dir, base_name + '.geojson')

    current_id = 1
    finished = [False]

    poly_lines = []
    point_scatters = []
    text_labels = []
    patches = []
    status_text = [None]

    base_points = []
    top_points = []
    cuboids = []

    has_template = [False]
    is_dragging = [False]
    drag_point_idx = [-1]
    drag_threshold = 15

    # Динамический вектор сдвига
    offset_vector = [0, 0]

    # Размер шаблона по умолчанию (в px)
    DEFAULT_WIDTH = 60
    DEFAULT_HEIGHT = 40
    DEFAULT_DEPTH = 25

    def update_status():
        if status_text[0]:
            status_text[0].remove()

        if not has_template[0]:
            progress = "Начать - ЛКМ"
        else:
            height_px = int((offset_vector[0]**2 + offset_vector[1]**2)**0.5)
            progress = f"Тяни угол | Высота: {height_px} px"

        txt = ax.text(0.02, 0.98,
                      f'Готово кубоидов: {len(cuboids)} | Следующий ID: {current_id}\n{progress}',
                      transform=ax.transAxes, fontsize=12, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        status_text[0] = txt

    # Сформировать GeoJSON и JSON ==============================================
    def save_to_geojson():
        features = []
        simple_json_list = []

        for cuboid in cuboids:
            base = cuboid['base']
            top = cuboid['top']
            all_points = base + top

            dx = top[0][0] - base[0][0]
            dy = top[0][1] - base[0][1]
            height_px = int((dx**2 + dy**2)**0.5)

            features.append({
                "type": "Feature",
                "properties": {
                    "id": cuboid['id'],
                    "type": "cuboid",
                    "base_points": 4,
                    "top_points": 4,
                    "height_pixels": height_px
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [base + [base[0]]]
                },
                "cuboid_3d": {
                    "base": base,
                    "top": top,
                    "all_8_points": all_points,
                    "offset_vector": [dx, dy]
                }
            })

            simple_json_list.append({
                "id": cuboid['id'],
                "base": base,
                "top": top,
                "height_pixels": height_px,
                "offset_vector": [dx, dy]
            })

        geojson = {"type": "FeatureCollection", "features": features}

        # Сохранить GeoJSON на комп
        with open(geojson_filename, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)
        print(f"\nСохранено в Drive: {geojson_filename}")

        # Сохранить JSON на комп
        json_filename = os.path.splitext(geojson_filename)[0] + '.json'
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(simple_json_list, f, indent=2, ensure_ascii=False)
        print(f"Сохранено в Drive (JSON): {json_filename}")

        download_path_geo = f"/content/{base_name}.geojson"
        with open(download_path_geo, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)
        files.download(download_path_geo)

        download_path_json = f"/content/{base_name}.json"
        with open(download_path_json, 'w', encoding='utf-8') as f:
            json.dump(simple_json_list, f, indent=2, ensure_ascii=False)
        files.download(download_path_json)

        print(f"Файлы скачаны: {base_name}.geojson и {base_name}.json")
        print(f"Всего кубоидов: {len(features)}")


    def create_template(cx, cy):
        """Создаeт шаблон кубоида вокруг точки клика"""
        nonlocal offset_vector

        base_points.append([int(cx - DEFAULT_WIDTH/2), int(cy + DEFAULT_HEIGHT/2)])
        base_points.append([int(cx + DEFAULT_WIDTH/2), int(cy + DEFAULT_HEIGHT/2)])
        base_points.append([int(cx + DEFAULT_WIDTH/2), int(cy - DEFAULT_HEIGHT/2)])
        base_points.append([int(cx - DEFAULT_WIDTH/2), int(cy - DEFAULT_HEIGHT/2)])

        offset_vector[0] = 0
        offset_vector[1] = -DEFAULT_DEPTH

        top_points.append([int(cx - DEFAULT_WIDTH/2), int(cy + DEFAULT_HEIGHT/2 - DEFAULT_DEPTH)])
        top_points.append([int(cx + DEFAULT_WIDTH/2), int(cy + DEFAULT_HEIGHT/2 - DEFAULT_DEPTH)])
        top_points.append([int(cx + DEFAULT_WIDTH/2), int(cy - DEFAULT_HEIGHT/2 - DEFAULT_DEPTH)])
        top_points.append([int(cx - DEFAULT_WIDTH/2), int(cy - DEFAULT_HEIGHT/2 - DEFAULT_DEPTH)])

        has_template[0] = True

    def get_point_under_cursor(x, y):
        for i, pt in enumerate(base_points):
            dist = ((pt[0] - x)**2 + (pt[1] - y)**2)**0.5
            if dist < drag_threshold:
                return i

        if top_points:
            for i, pt in enumerate(top_points):
                dist = ((pt[0] - x)**2 + (pt[1] - y)**2)**0.5
                if dist < drag_threshold:
                    return i + 4

        return -1

    def redraw():
        for line in poly_lines: line.remove()
        for scat in point_scatters: scat.remove()
        for txt in text_labels: txt.remove()
        for patch in patches: patch.remove()
        poly_lines.clear()
        point_scatters.clear()
        text_labels.clear()
        patches.clear()

        for cuboid in cuboids:
            base = cuboid['base']
            top = cuboid['top']

            # Нижняя плоскость (без заливки, черный пунктир)
            base_closed = base + [base[0]]
            xs, ys = zip(*base_closed)
            line = ax.plot(xs, ys,
                           'black',
                           linewidth=0.8,
                           linestyle='--')[0]
            poly_lines.append(line)

            # Верхняя плоскость
            top_closed = top + [top[0]]
            xs, ys = zip(*top_closed)
            line = ax.plot(xs, ys,
                           'gray',
                           linewidth=1)[0]
            poly_lines.append(line)

            # Заливка верхней плоскости
            poly_top = plt.Polygon(top,
                                   closed=True,
                                   facecolor='lime',
                                   alpha=0.7,
                                   edgecolor='none')
            ax.add_patch(poly_top)
            patches.append(poly_top)

            # Боковые вертикальные ребра
            for i in range(4):
                xs = [base[i][0], top[i][0]]
                ys = [base[i][1], top[i][1]]
                line = ax.plot(xs, ys,
                               'gray',
                               linewidth=1)[0]
                poly_lines.append(line)

                # Заливка боковых плоскостей
                side_pts = [base[i], base[(i+1)%4], top[(i+1)%4], top[i]]
                poly_side = plt.Polygon(side_pts,
                                        closed=True,
                                        facecolor='lime',
                                        alpha=0.4,
                                        edgecolor='none')
                ax.add_patch(poly_side)
                patches.append(poly_side)

            # ID кубоида с надписью
            txt = ax.text(base[0][0] + 13, base[0][1] - 5, str(cuboid['id']),
                          color='orange',
                          fontsize=10,
                          zorder=10)
            text_labels.append(txt)

        # АКТИВНЫЙ ШАБЛОН (который еще не завершен)
        if has_template[0] and base_points:
            scat = ax.scatter([p[0] for p in base_points], [p[1] for p in base_points],
                              c='red',
                              s=5,
                              zorder=5)
            point_scatters.append(scat)

            # Замкнутый контур базы (работает при любом количестве точек >=2)
            if len(base_points) >= 2:
                closed_base = base_points + [base_points[0]]
                line = ax.plot([p[0] for p in closed_base], [p[1] for p in closed_base],
                               'red',
                               linewidth=1,
                               linestyle='--')[0]
                poly_lines.append(line)

            if top_points:
                scat = ax.scatter([p[0] for p in top_points], [p[1] for p in top_points],
                                  c='blue',
                                  s=5,
                                  zorder=6)
                point_scatters.append(scat)

                closed_top = top_points + [top_points[0]]
                poly_lines.append(ax.plot([p[0] for p in closed_top], [p[1] for p in closed_top],
                                          'blue',
                                          linewidth=1)[0])

                for i in range(min(len(base_points), len(top_points))):
                    poly_lines.append(ax.plot([base_points[i][0], top_points[i][0]],
                                              [base_points[i][1], top_points[i][1]],
                                              'blue',
                                              linewidth=1)[0])

        update_status()
        fig.canvas.draw_idle()

    # События на кликах ========================================================
    def on_click(event):
        nonlocal current_id, base_points, top_points, has_template, offset_vector
        if finished[0]:
            return
        if event.inaxes != ax:
            return
        if len(cuboids) >= max_spaces:
            print(f"Достигнут максимум мест: {max_spaces}")
            return

        # ПРАВЫЙ КЛИК (button=3) — финализация кубоида
        if event.button == 3:
            if has_template[0] and base_points and top_points:
                cuboids.append({
                    'id': current_id,
                    'base': base_points.copy(),
                    'top': top_points.copy()
                })
                print(f"Кубоид #{current_id} завершен (ПКМ)")

                base_points.clear()
                top_points.clear()
                offset_vector[0] = 0
                offset_vector[1] = 0
                has_template[0] = False
                current_id += 1
                redraw()
            else:
                print("Создать кубоид")
            return

        # ЛЕВЫЙ КЛИК (button=1) — создать шаблон
        if event.button == 1:
            if has_template[0]:
                return

            x, y = int(event.xdata), int(event.ydata)
            create_template(x, y)
            print(f"Шаблон создан в ({x}, {y})")
            redraw()

    def on_press(event):
        nonlocal is_dragging, drag_point_idx
        if finished[0] or event.inaxes != ax:
            return
        if not has_template[0]:
            return

        x, y = event.xdata, event.ydata
        point_idx = get_point_under_cursor(x, y)

        if point_idx >= 0:
            is_dragging[0] = True
            drag_point_idx[0] = point_idx
            if point_idx <= 3:
                print(f"Захват красного угла #{point_idx}")
            else:
                print(f"Захват синего угла #{point_idx - 4} (можно менять высоту)")

    def on_motion(event):
        nonlocal offset_vector
        if finished[0] or event.inaxes != ax:
            return
        if not is_dragging[0]:
            return

        x, y = int(event.xdata), int(event.ydata)

        # Тянем красный угол - верх следует с текущим offset
        if 0 <= drag_point_idx[0] <= 3:
            base_points[drag_point_idx[0]] = [x, y]
            top_points[drag_point_idx[0]] = [x + offset_vector[0], y + offset_vector[1]]

        # Тянем синий угол - обновится offset_vector для всех углов
        elif 4 <= drag_point_idx[0] <= 7:
            idx = drag_point_idx[0] - 4
            top_points[idx] = [x, y]

            # Вычисляем новый вектор сдвига из этой пары
            new_dx = x - base_points[idx][0]
            new_dy = y - base_points[idx][1]

            # Применяем новый offset ко всем 4 углам сразу
            offset_vector[0] = new_dx
            offset_vector[1] = new_dy

            for i in range(4):
                top_points[i] = [base_points[i][0] + offset_vector[0],
                                 base_points[i][1] + offset_vector[1]]

        redraw()

    def on_release(event):
        nonlocal is_dragging, drag_point_idx
        is_dragging[0] = False
        drag_point_idx[0] = -1

    # Клавиши событий ==========================================================
    def on_key_press(event):
        nonlocal current_id, base_points, top_points, has_template, offset_vector
        print(f"Нажата клавиша: '{event.key}'")

        # КЛАВИША '1' или 's' — сохранить и выйти (без mpl_disconnect)
        if event.key in ('1', 's', 'S'):
            print(f"\nСохранение в: {geojson_filename}")
            print(f"Всего кубоидов: {len(cuboids)}")
            save_to_geojson()
            finished[0] = True
            # Без mpl_disconnect(None) — с ним будет TypeError
            print("Все события заблокированы. Закрыть ячейку вручную.")
            return

        # КЛАВИША '2' или 'd' — начать новый кубоид
        elif event.key in ('2', 'd', 'D'):
            if has_template[0] and base_points and top_points:
                cuboids.append({
                    'id': current_id,
                    'base': base_points.copy(),
                    'top': top_points.copy()
                })
                print(f"Кубоид #{current_id} завершен (клавиша 2)")

                base_points.clear()
                top_points.clear()
                offset_vector[0] = 0
                offset_vector[1] = 0
                has_template[0] = False
                current_id += 1
                redraw()
            else:
                print("Сначала создайте и настройте кубоид, затем нажмите '2'")

        elif event.key.lower() == 'q':
            print("Выход без сохранения")
            finished[0] = True
            # без mpl_disconnect
            print("Все события заблокированы. Закрыть ячейку вручную.")
            return

        elif event.key.lower() == 'c':
            base_points.clear()
            top_points.clear()
            offset_vector[0] = 0
            offset_vector[1] = 0
            has_template[0] = False
            print("Шаблон отменён")
            redraw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    update_status()
    plt.show()

# Запуск функции
annotate_cuboids_cvat_style(
    image_path='/content/drive/MyDrive/PARKING/dataset/images/train/2013-01-29_14_46_16.jpg',
    save_dir='/content/drive/MyDrive/PARKING/POINT_100/Annotations',
    max_spaces=100
)


# КЛАВИШИ:

# ЛКМ - появится разверстка кубоида
# сначала тащить красные точки нижей плоскости (база), потом тащитьсиние точки и строить верх кубоида
# ПКМ - финализация редактуры расположения точек (станет лаймовым)
# Развилка: если разметка 1 шт кубоида - клавиша 1 (конец, сохранение инфо в файлы --> на диск компа и в content/ )
#           если размечать другие - клавиша 2 (потом закончить 1)
# Q - Выйти без сохранения
# C - Отменить текущий шаблон