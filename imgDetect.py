import cv2
import numpy as np
import sys

# Завантаження конфігураційного файлу та вагових файлів YOLO
config_path = 'yolo/yolov3.cfg'
weights_path = 'yolo/yolov3.weights'
net = cv2.dnn.readNet(weights_path, config_path)

# Завантаження назв об'єктів, які YOLO може розпізнати
classes = []
with open('yolo/coco.names', 'r') as f:
    classes = f.read().splitlines()

# Завантаження зображення
try:
    image_path = sys.argv[1]
except IndexError:
    print('Usage: python imgDetect.py <image_path>')
#image_path = 'img/mercedes.jpg'
image = cv2.imread(image_path)
height, width, _ = image.shape

# Підготовка зображення для подачі в мережу YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# Отримання розпізнаних об'єктів
outs = net.forward(net.getUnconnectedOutLayersNames())

# Визначення параметрів для NMS
conf_threshold = 0.6
nms_threshold = 0.4

# Збереження координат рамок та ймовірностей для NMS
boxes = []
confidences = []
class_ids = []

# Виведення результатів розпізнавання
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Обчислення координат кутів прямокутника
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Збереження координат рамок та ймовірностей
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)  # Збереження id класу

# Використання non-maximum suppression для видалення дубльованих рамок
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Відображення результатів
for i in indices:
    box = boxes[i]  # Отримати координати рамки
    x, y, w, h = box
    class_id = class_ids[i]  # Отримати id класу для поточного об'єкта

    # Відображення прямокутника та назви об'єкта
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(image, classes[class_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Відображення зображення з розпізнаними об'єктами
cv2.imshow('YOLO Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
