# Разметка кубоидов без CVAT
Учитывая, что CVAT заблокирован в веб-версии, можно поднимать его через docker. 
Но можно обойтись без спецфреймворков и размечать изображения кодом на Python в ноутбуке, формируя там интерактивную среду.
Однако, если делать локально в Jupiter, то много "танцев с бубнами". Кроме того, возникает нагрузка рендеринга графики на процессор. Гораздо проще и легче по затратам решать задачу в облачном colab.
Здесь приведен пример разметки кубоидов (= псевдо-3D). Код для разметки в 2D (ббоксов, полигонов), разумеется граздо проще.

## Cuboid Annotation Without CVAT
Although the web version of CVAT is currently unavailable, it can be self-hosted via Docker. That said, you can skip dedicated annotation frameworks altogether and label images directly on your laptop using Python in an interactive environment.
Running this locally in Jupyter typically requires a lot of configuration tweaks and can place a heavy rendering load on your CPU. Using Google Colab in the cloud is a much simpler and more resource-efficient alternative.
The example below demonstrates cuboid annotation (pseudo-3D). For reference, annotating standard 2D objects (bounding boxes, polygons) is significantly more straightforward.


## Как делается:
- Происходит обработка событий (все откомментировано)
- Углы кубоида перетаскиваются мышкой - важно точно отрисовать нижнюю плоскость базы
- При построении верхних над базой плоскостей происходит авторасчет высоты и смещения
- Каждый кубоид получает уникальный ID
- Результаты записываются в формате GeoJSON и JSON и автоматически сохраняются на комп

## How It Works
- **Event handling**: All UI events are processed and fully commented in the code.
- **Interactive dragging**: Drag cuboid corners with the mouse. Accurately positioning the bottom base face is essential.
- **Auto-calculation**: Height and offset are automatically computed once the top faces are defined relative to the base.
- **Unique IDs**: Each cuboid is assigned a unique identifier.
- **Auto-save**: Results are automatically saved locally in both GeoJSON and JSON formats.

## Markup example
![screenshot_Parking_Cuboids](https://github.com/user-attachments/assets/56f187cb-c943-4804-822e-d1019d3c5921)


## Project structure
- `annotation_cuboids' — the code in the file .py - insert into Google Colaboratory 
- `2013-01-29_14_46_16.json` is a file in the json format. 
- `assets/` — screenshot of the result of working with the code
