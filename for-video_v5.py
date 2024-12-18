import cv2
import os
from ultralytics import YOLO
import numpy as np

# Загрузка предобученной модели YOLOv5 для сегментации
version='5'
model = YOLO('./yolov5xu.pt')  # Укажите путь к модели YOLOv5

# Путь к видеофайлу
video_path = "video/geo/v00000.mkv"  # Укажите путь к вашему видеофайлу

# Папка для сохранения результатов
output_folder = f"output_images_{os.path.basename(video_path)}_{version}/"
os.makedirs(output_folder, exist_ok=True)

# Открытие видео
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Ошибка при открытии видео")
    exit()

# Получаем параметры видео
video_fps = cap.get(cv2.CAP_PROP_FPS)  # Частота кадров
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Ширина кадра
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Высота кадра
frame_interval = int(video_fps)  # 1 кадр в секунду

frame_idx = 0  # Индекс текущего кадра

# Список классов COCO
coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", 
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Обработка видео
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break  # Конец видео

    # Обрабатываем каждый кадр, соответствующий 1 кадру в секунду
    if frame_idx % frame_interval == 0:
        # YOLO сегментация
        results = model(frame)

        # Проверка, есть ли детекции
        if results[0].boxes is not None:
            classes = results[0].boxes.cls.numpy()  # Классы объектов
            boxes = results[0].boxes.xyxy.numpy()  # Координаты объектов

            # Рисуем bounding box и текст с правильными именами классов
            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box)
                label = coco_classes[int(cls)]  # Получаем имя класса по индексу
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Сохраняем результат как изображение
            output_path = os.path.join(output_folder, f"frame_{frame_idx}.jpg")
            cv2.imwrite(output_path, frame)

    frame_idx += 1

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()