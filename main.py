from ultralytics import YOLO
import cv2


model = YOLO("yolov8x.pt")
image_path = "/home/ozma/Documents/Datasets/saffron/Unlabeled/002.jpg"
results = model.predict(source="0", show=True)
cv2.waitKey()