import numpy as np

def xyxy_to_xywh(xyxy):
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2
    yc = y1 + h / 2
    return [xc, yc, w, h]

def cosine_similarity(a, b):
    """Compute cosine similarity between two feature vectors."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
