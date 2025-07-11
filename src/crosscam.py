import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from torch.nn.functional import cosine_similarity
import os

# Load YOLO model
detector = YOLO("models/yolov11.pt") 

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model.eval()


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 64)),
    transforms.ToTensor(),
])

# Extracting embeddings from detections
def extract_embeddings(frame, boxes):
    embeddings = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        input_tensor = transform(crop).unsqueeze(0)
        with torch.no_grad():
            embedding = model(input_tensor)
            embeddings.append(embedding.squeeze(0))
    return embeddings

# similarity matrix
def match_embeddings(emb1, emb2):
    scores = torch.zeros((len(emb1), len(emb2)))
    for i, e1 in enumerate(emb1):
        for j, e2 in enumerate(emb2):
            scores[i][j] = cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
    return scores

cap_tacticam = cv2.VideoCapture("videos/tacticam.mp4")
cap_broadcast = cv2.VideoCapture("videos/broadcast.mp4")

output_path = "videos/output.mp4"
frame_width = int(cap_tacticam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap_tacticam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap_tacticam.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frame_idx = 0
while cap_tacticam.isOpened() and cap_broadcast.isOpened():
    ret1, frame_tacticam = cap_tacticam.read()
    ret2, frame_broadcast = cap_broadcast.read()
    if not ret1 or not ret2:
        break

    # Detect players
    results_tacticam = detector(frame_tacticam)[0]
    results_broadcast = detector(frame_broadcast)[0]

    tacticam_boxes = results_tacticam.boxes.xyxy.cpu().numpy() if results_tacticam.boxes else []
    broadcast_boxes = results_broadcast.boxes.xyxy.cpu().numpy() if results_broadcast.boxes else []

    tacticam_embeddings = extract_embeddings(frame_tacticam, tacticam_boxes)
    broadcast_embeddings = extract_embeddings(frame_broadcast, broadcast_boxes)

    # Match players
    matches = []
    if tacticam_embeddings and broadcast_embeddings:
        score_matrix = match_embeddings(tacticam_embeddings, broadcast_embeddings)
        for i, row in enumerate(score_matrix):
            j = torch.argmax(row)
            sim = row[j].item()
            if sim > 0.7:
                matches.append((i, j.item(), sim))

    # bounding boxes and labels
    for i, box in enumerate(tacticam_boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame_tacticam, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_tacticam, f"T{i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for i, box in enumerate(broadcast_boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame_broadcast, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame_broadcast, f"B{i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    for tidx, bidx, sim in matches:
        x1, y1, x2, y2 = map(int, tacticam_boxes[tidx])
        label = f"↔ B{bidx} ({sim:.2f})"
        cv2.putText(frame_tacticam, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if frame_idx == 0:
        print("🔁 Cross-Camera Matches:")
        for tidx, bidx, sim in matches:
            print(f"Tacticam Player {tidx} ↔ Broadcast Player {bidx} | Similarity: {sim:.2f}")

    out.write(frame_tacticam)
    frame_idx += 1

cap_tacticam.release()
cap_broadcast.release()
out.release()
cv2.destroyAllWindows()
