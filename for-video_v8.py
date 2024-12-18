import cv2
import os
from ultralytics import YOLO
import numpy as np

# Загрузка предобученной модели YOLOv8
version = '8'
model = YOLO('yolov8x.pt')  # Используем предобученную модель YOLOv8 для обнаружения объектов

# Словарь классов для COCO
coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "none", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "none", "backpack", "umbrella", "none", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "none", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "none", "toilet", "none", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

video_path = "video/video1.MP4"  # Укажите путь к вашему видеофайлу

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

# Получаем исходные параметры видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Ширина кадра
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Высота кадра
video_fps = cap.get(cv2.CAP_PROP_FPS)  # Частота кадров в исходном видео
frame_interval = int(video_fps // 1)  # Частота кадров для извлечения (каждую секунду)

frame_idx = 0  # Индекс текущего кадра

# Списки для хранения значений IoU
ious = []

# Обработка видео
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break  # Завершаем, если достигнут конец видео

    # Пропускаем кадры, чтобы достичь 1 кадра в секунду
    if frame_idx % frame_interval == 0:
        # YOLO обнаружение объектов
        results = model(frame)

        # Проверка, есть ли детекции
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.numpy()  # Координаты объектов
            classes = results[0].boxes.cls.numpy()  # Классы объектов

            # Рисуем bounding box и текст с правильными именами классов
            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box)
                label = coco_classes[int(cls)]  # Получаем имя класса по индексу
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Пример: Для вычисления IoU сравниваем с фиксированным ground truth (например, для объекта)
                ground_truth_box = [100, 100, 200, 200]  # Пример ground truth
                iou = calculate_iou(box, ground_truth_box)
                ious.append(iou)

            # Сохраняем результат как изображение
            output_path = os.path.join(output_folder, f"frame_{frame_idx}.jpg")
            cv2.imwrite(output_path, frame)

    frame_idx += 1

# Вычисляем среднее IoU (MOI)
mean_iou = np.mean(ious) if ious else 0
print(f"Mean IoU (MOI): {mean_iou}")

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
