import cv2
import os
from ultralytics import YOLO
import numpy as np
from sklearn.cluster import KMeans

# Загрузка предобученной модели YOLOv8
version = '8'
model = YOLO('yolov8x.pt')  # Используем предобученную модель YOLOv8 для обнаружения объектов

# Словарь классов для COCO
coco_classes = coco_classes = [
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

# Папка для сохранения результатов
output_folder = f"output_images_{os.path.basename(video_path)}_{version}/"
os.makedirs(output_folder, exist_ok=True)

# Функция для вычисления IoU
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    # Координаты пересечения
    x_left = max(x1, x1_gt)
    y_top = max(y1, y1_gt)
    x_right = min(x2, x2_gt)
    y_bottom = min(y2, y2_gt)

    # Площадь пересечения
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Площадь объединения
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union_area = box1_area + box2_area - intersection_area

    # IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

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

# Метрики
ious = []
confidences = []
previous_boxes = None  # Для сравнения между кадрами

# Открываем файл для записи метрик
metrics_file = output_folder+"metrics_output.txt"
with open(metrics_file, "w", encoding="utf-8-sig") as f:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            results = model(frame)

            if results[0].boxes is not None and len(results[0].boxes) > 0:  # Проверяем наличие детекций
                boxes = results[0].boxes.xyxy.numpy()
                current_confidences = results[0].boxes.conf.numpy()
                current_classes = results[0].boxes.cls.numpy()
                current_boxes = [list(map(int, box)) for box in boxes]
                confidences.extend(current_confidences)

                # Запись информации о кадре
                frame_name = f"frame_{frame_idx}.jpg"
                f.write(f"Кадр: {frame_name}\n")

                # Кластеризация предсказаний
                centroids = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in current_boxes]
                if len(centroids) >= 3:  # Минимальное число кластеров
                    kmeans = KMeans(n_clusters=3).fit(centroids)
                    f.write(f"  Кластеры: {kmeans.labels_.tolist()}\n")

                # Сравнение IoU с предыдущим кадром
                if previous_boxes is not None:
                    for box1 in current_boxes:
                        for box2 in previous_boxes:
                            iou = calculate_iou(box1, box2)
                            ious.append(iou)

                previous_boxes = current_boxes

                # Визуализация предсказаний
                for box, cls, conf in zip(boxes, current_classes, current_confidences):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = coco_classes[int(cls)]
                    label = f"{class_name}: {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Запись метрик объекта
                    f.write(f"  Объект: {class_name}, Уверенность: {conf:.2f}, Координаты: ({x1}, {y1}), ({x2}, {y2})\n")

                # Сохранение кадра
                output_path = os.path.join(output_folder, frame_name)
                cv2.imwrite(output_path, frame)

                f.write("\n")  # Разделитель для следующего кадра

        frame_idx += 1

# Рассчитываем метрики
mean_confidence = np.mean(confidences) if confidences else 0
mean_iou = np.mean(ious) if ious else 0

# Записываем метрики в файл
with open(metrics_file, "a") as f:
    f.write(f"\nMean_confidence: {mean_confidence:.2f}\n")
    f.write(f"Mean_iou: {mean_iou:.2f}\n")

# Завершаем работу
cap.release()
cv2.destroyAllWindows()
print(f"Метрики записаны в файл {metrics_file}")
