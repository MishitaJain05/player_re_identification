from ultralytics import YOLO
import cv2
import os


model_path = os.path.join("models", "yolov11.pt")
video_path = os.path.join("videos", "15sec_input_720p.mp4")
output_path = os.path.join("output", "detections.avi")


model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)
width, height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)[0]
    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        if conf > 0.5 and int(cls) == 0:  # class 0 = player
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
    out.write(frame)
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
