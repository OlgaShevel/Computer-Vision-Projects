import os, json, numpy as np, cv2
import matplotlib.pyplot as plt
from matplotlib.path import Path
import onnxruntime as ort
import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_DIR, 'best_parking_100spots.onnx')
CALIBRATE_PATH = os.path.join(PROJECT_DIR, 'calibrate_camera_01.json')
GEOJSON_DIR = os.path.join(PROJECT_DIR, 'ALL_100_geojson')
TEST_DIR = os.path.join(PROJECT_DIR, 'test')

session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

park_numbers = sorted([int(f.split('_')[1].split('.')[0]) for f in os.listdir(GEOJSON_DIR) if f.endswith('.geojson')])
test_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.jpg')])

def process_parking(park_idx, test_filename):
    try:
        plt.switch_backend('Agg')
        
        with open(CALIBRATE_PATH, 'r') as f:
            cal_data = json.load(f)
        # pts = cal_data.get('points', [])
        # px = np.array([[p[0], p[1]] for p in pts], dtype=np.float32).reshape(-1, 1, 2)
        # wd = np.array([[p[2], p[3]] for p in pts], dtype=np.float32).reshape(-1, 1, 2)
        # H, _ = cv2.findHomography(px, wd)

        geo_path = os.path.join(GEOJSON_DIR, f'park_{park_idx}.geojson')
        with open(geo_path, 'r') as f:
            geo = json.load(f)
        coords = geo['features'][0]['geometry']['coordinates'][0]
        spot_poly = Path(coords)

        img_path = os.path.join(TEST_DIR, test_filename)
        img = cv2.imread(img_path)
        if img is None:
            return None, f"Не удалось загрузить изображение: {test_filename}"
            
        orig_h, orig_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        IMG_SIZE = 1280
        scale = min(IMG_SIZE / orig_w, IMG_SIZE / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        dx, dy = (IMG_SIZE - new_w) // 2, (IMG_SIZE - new_h) // 2
        canvas[dy:dy+new_h, dx:dx+new_w] = resized
        inp = (canvas[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
        
        outs = session.run(None, {session.get_inputs()[0].name: inp})[0]

        occupied = False
        occupied_bbox = None
        for i in range(outs.shape[2]):
            conf = outs[0, 4, i]
            if conf >= 0.55:
                xc, yc, w, h = outs[0, :4, i]
                x1 = ((xc - w/2) - dx) / scale
                y1 = ((yc - h/2) - dy) / scale
                x2 = ((xc + w/2) - dx) / scale
                y2 = ((yc + h/2) - dy) / scale
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if spot_poly.contains_point((cx, cy)):
                    occupied = True
                    occupied_bbox = [x1, y1, x2, y2]
                    break

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img_rgb)
        ax.axis('off')
        
        if occupied_bbox:
            x1, y1, x2, y2 = map(int, occupied_bbox)
            ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none'))
            
        poly_color = 'red' if occupied else 'lime'
        ax.add_patch(plt.Polygon(coords, closed=True, linewidth=4, edgecolor=poly_color, facecolor=poly_color, alpha=0.3))
        
        label_y = min([p[1] for p in coords])
        ax.text(np.mean([p[0] for p in coords]), label_y - 30, f'{park_idx}', 
                color='black', fontsize=10, weight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', edgecolor='gray', alpha=0.9))
        
        status_text = "true (ЗАНЯТО)" if occupied else "false (СВОБОДНО)"
        ax.set_title(f'Место № {park_idx} | {status_text}', fontsize=12, pad=10)
        plt.tight_layout()

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_out = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        plt.close(fig)

        return img_out, status_text
    except Exception as e:
        return None, str(e)

demo = gr.Interface(
    fn=process_parking,
    inputs=[
        gr.Dropdown(choices=park_numbers, label="Номер машиноместа", value=park_numbers[0] if park_numbers else None),
        gr.Dropdown(choices=test_files, label="Тестовое изображение", value=test_files[0] if test_files else None)
    ],
    outputs=[
        gr.Image(label="Результат детекции"),
        gr.Textbox(label="Статус")
    ],
    title="Parking Spot Detector Demo",
    description="Выберите номер места и фото, нажмите Submit. Визуализация и статус генерируются ONNX-моделью с учётом гомографии."
)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)