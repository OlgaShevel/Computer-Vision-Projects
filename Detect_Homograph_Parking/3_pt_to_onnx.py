# Конвертировать .pt в onnx для imgsz=1280

from ultralytics import YOLO

model = YOLO('/content/drive/MyDrive/PARKING/POINT_100/MODELS/best_parking_100spots.pt')


model.export(
    format='onnx',
    imgsz=1280,
    simplify=True,
    opset=12,
    half=False
)
