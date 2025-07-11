from sort import Sort  
from ultralytics import YOLO
import cv2
import os



model = YOLO("models/yolov11.pt")
cap = cv2.VideoCapture("videos/15sec_input_720p.mp4")
tracker = Sort()
out = cv2.VideoWriter("output/sort_tracked.avi", cv2.VideoWriter_fourcc(*'XVID'), 20, (int(cap.get(3)), int(cap.get(4))))



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = []
    results = model(frame)[0]
    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        if int(cls) == 0:
            detections.append([x1, y1, x2, y2, conf])
    dets = tracker.update(np.array(detections))
    
    for *xyxy, id in dets:
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {int(id)}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    out.write(frame)
    cv2.imshow("Track", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
