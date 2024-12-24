import cv2
import os
from ultralytics import YOLO
import numpy as np

# Загрузка предобученной модели YOLOv8 для сегментации
version = '8'
model = YOLO('yolov8x-seg.pt')  # Используем сегментационную модель YOLOv8

# Словарь классов для COCO
coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

video_path = "video/video1.mp4"  # Укажите путь к вашему видеофайлу

# Папки для сохранения результатов
output_folder = f"output_images_{os.path.basename(video_path)}_{version}_instanse/"
mask_output_folder = output_folder+"mask_output/"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(mask_output_folder, exist_ok=True)

# Функция для наложения маски на изображение
def apply_mask(image, mask, color):
    """Наложение маски на изображение."""
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))  # Приведение маски к размеру изображения
    mask = mask > 0.5  # Бинаризация маски
    for c in range(3):  # Применение цвета ко всем каналам (RGB)
        image[:, :, c] = np.where(mask, color[c], image[:, :, c])

# Функция для сохранения маски
def save_mask(mask, output_path):
    mask = (cv2.resize(mask, (640, 480)) * 255).astype(np.uint8)  # Преобразование в формат изображения
    cv2.imwrite(output_path, mask)

# Открытие видео
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Ошибка при открытии видео")
    exit()

# Получаем параметры видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(video_fps // 1)

frame_idx = 0

# Открываем файл для записи метрик
metrics_file = output_folder + "metrics_output.txt"
with open(metrics_file, "w", encoding="utf-8-sig") as f:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            results = model(frame)

            if results[0].masks is not None:  # Проверяем наличие масок
                masks = results[0].masks.data.cpu().numpy()  # Маски объектов
                boxes = results[0].boxes.xyxy.numpy()  # Координаты рамок
                current_confidences = results[0].boxes.conf.numpy()
                current_classes = results[0].boxes.cls.numpy()

                # Запись информации о кадре
                frame_name = f"frame_{frame_idx}.jpg"
                f.write(f"Кадр: {frame_name}\n")

                # Наложение масок и запись метрик
                for i, (mask, box, cls, conf) in enumerate(zip(masks, boxes, current_classes, current_confidences)):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = coco_classes[int(cls)]
                    label = f"{class_name}: {conf:.2f}"

                    # Генерация случайного цвета для маски
                    color = np.random.randint(0, 255, (3,), dtype=int)

                    # Масштабирование маски и наложение её на кадр
                    apply_mask(frame, mask, color)

                    # Рисуем рамку и подпись
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color.tolist(), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

                    # Сохранение маски как отдельного файла
                    mask_filename = f"frame_{frame_idx}_mask_{i}_{class_name}.png"
                    mask_output_path = os.path.join(mask_output_folder, mask_filename)
                    save_mask(mask, mask_output_path)

                    # Запись метрик маски
                    f.write(f"  Маска {i}: {class_name}, Уверенность: {conf:.2f}, Координаты: ({x1}, {y1}), ({x2}, {y2}), Путь к маске: {mask_output_path}\n")

                # Сохранение кадра с наложенными масками
                output_path = os.path.join(output_folder, frame_name)
                cv2.imwrite(output_path, frame)

                f.write("\n")  # Разделитель для следующего кадра

        frame_idx += 1

# Завершаем работу
cap.release()
cv2.destroyAllWindows()
print(f"Метрики записаны в файл {metrics_file}")
