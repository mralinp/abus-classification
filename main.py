from ultralytics import YOLO
import cv2


model = YOLO("yolov8x.pt")
image_path = "/home/ozma/Documents/Datasets/saffron/Unlabeled/002.jpg"
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    results = model.predict(source=frame, show=True)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()