import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Starting inference... press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=512, conf=0.4, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("Grape Disease Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()