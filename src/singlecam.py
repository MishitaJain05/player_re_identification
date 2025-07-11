import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import xyxy_to_xywh


model = YOLO("models/yolov11.pt")
cap = cv2.VideoCapture("videos/15sec_input_720p.mp4")
w, h = int(cap.get(3)), int(cap.get(4))


tracker = DeepSort(max_age=15, n_init=2, max_cosine_distance=0.4)

out = cv2.VideoWriter("output/singlecam_result.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      20, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    dets = []
    results = model(frame)[0]
    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        if int(cls) != 0 or conf < 0.5:
            continue
        xywh = xyxy_to_xywh([x1, y1, x2, y2])
        dets.append((xywh, conf, "player"))

    tracks = tracker.update_tracks(dets, frame=frame)

    for t in tracks:
        if not t.is_confirmed():
            continue
        tid = t.track_id
        l, t_, r, b = map(int, t.to_ltrb())
        cv2.rectangle(frame, (l, t_), (r, b), (255, 0, 0), 2)
        cv2.putText(frame, f"ID:{tid}", (l, t_ - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    out.write(frame)
    cv2.imshow("Single-Cam Re-ID", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
