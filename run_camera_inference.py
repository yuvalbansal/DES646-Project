import os
import json
import time
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

# ======== Config (tweak here or via CLI if you like) ========
CKPT_PATH = "runs/checkpoints/best_snapshot.pt"
META_PATH = "runs/checkpoints/metadata.json"
CAMERA_INDEX = 0                 # change to your USB cam index if needed
IMG_SIZE_DEFAULT = 224
PREDICTION_WINDOW = 8            # frames for smoothing
STABILITY_FRAMES = 5             # frames of the same top class before commit
MIN_CONFIDENCE = 0.70            # softmax prob threshold to commit
ROI_MARGIN = 20                  # extra pixels around detected hand box
DRAW = True                      # draw overlays
MIRROR = True                    # mirror camera for natural UX

# ======== Optional: try to use MediaPipe Hands for better ROI ========
try:
    import mediapipe as mp
    MP_AVAILABLE = True
    mp_hands = mp.solutions.hands
except Exception:
    MP_AVAILABLE = False
    mp_hands = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_metadata(meta_path):
    meta = {}
    if Path(meta_path).exists():
        meta = json.loads(Path(meta_path).read_text())
    return meta


def build_model(num_classes, img_size, ckpt):
    # Rebuild MobileNetV3-Small to match training
    model = models.mobilenet_v3_small(pretrained=False)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    sd = ckpt["model_state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval().to(DEVICE)
    return model


def get_preprocess(img_size, mean, std):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def softmax_np(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def get_hand_roi(frame_bgr, last_roi=None):
    """
    Returns: roi_bgr, (x1, y1, x2, y2) bbox
    Uses MediaPipe if available; otherwise returns a centered square crop.
    """
    h, w = frame_bgr.shape[:2]
    # Fallback: center square crop
    def center_crop():
        side = min(h, w)
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)

    if not MP_AVAILABLE:
        return center_crop()

    # Use MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        # NOTE: MediaPipe expects RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if not res.multi_hand_landmarks:
            # if we had a last ROI, keep using it briefly
            if last_roi is not None:
                x1, y1, x2, y2 = last_roi
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
                return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)
            return center_crop()

        # compute bbox around landmarks
        hand_landmarks = res.multi_hand_landmarks[0]
        xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
        ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
        x1, x2 = max(0, min(xs) - ROI_MARGIN), min(w, max(xs) + ROI_MARGIN)
        y1, y2 = max(0, min(ys) - ROI_MARGIN), min(h, max(ys) + ROI_MARGIN)
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return center_crop()
        return roi, (x1, y1, x2, y2)


def draw_overlay(frame, committed_text, current_pred, conf, bbox=None, fps=None):
    h, w = frame.shape[:2]
    y = 30
    # Committed text
    cv2.putText(frame, f"Text: {committed_text[-60:]}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    y += 30
    # Current prediction
    cv2.putText(frame, f"Current: {current_pred} ({conf:.2f})", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
    y += 30
    # FPS
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
    # BBox
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)


def main():
    # ---------- Load checkpoint & metadata ----------
    if not Path(CKPT_PATH).exists():
        raise FileNotFoundError(f"Model weights not found at {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    meta = load_metadata(META_PATH)
    class_names = ckpt.get("class_names", meta.get("class_names"))
    if class_names is None:
        raise RuntimeError("class_names not found in checkpoint or metadata.json")

    img_size = ckpt.get("img_size", meta.get("img_size", IMG_SIZE_DEFAULT))
    mean = ckpt.get("mean", meta.get("mean", [0.485, 0.456, 0.406]))
    std  = ckpt.get("std",  meta.get("std",  [0.229, 0.224, 0.225]))

    preprocess = get_preprocess(img_size, mean, std)
    model = build_model(num_classes=len(class_names), img_size=img_size, ckpt=ckpt)

    # ---------- Video capture ----------
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    committed_text = ""
    window = deque(maxlen=PREDICTION_WINDOW)  # (label_idx, prob)
    stable_counter = 0
    last_label_idx = None
    last_bbox = None

    prev_time = time.time()
    fps = 0.0

    print("Controls: [c] Clear | [b] Backspace | [q] Quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if MIRROR:
            frame = cv2.flip(frame, 1)

        roi_bgr, bbox = get_hand_roi(frame, last_bbox)
        last_bbox = bbox

        # Preprocess ROI for model
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        pil_like = cv2.resize(roi_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
        x = preprocess(pil_like).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs = probs[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = class_names[idx]

        # Smoothing window
        window.append((idx, conf))
        # Majority vote on indices in the window
        counts = Counter([i for i, _ in window])
        top_idx, occurrences = counts.most_common(1)[0]
        # Average confidence for that class in window
        avg_conf = np.mean([c for i, c in window if i == top_idx])
        top_label = class_names[top_idx]

        # Debounce & commit if stable
        if top_idx == last_label_idx and avg_conf >= MIN_CONFIDENCE:
            stable_counter += 1
        else:
            stable_counter = 1
            last_label_idx = top_idx

        committed_char = None
        if stable_counter >= STABILITY_FRAMES:
            # Commit only when the class is not 'nothing'
            if top_label == "space":
                committed_text += " "
                committed_char = " "
            elif top_label == "del":
                committed_text = committed_text[:-1]
                committed_char = "<DEL>"
            elif top_label != "nothing":
                committed_text += top_label
                committed_char = top_label
            stable_counter = 0  # reset after commit

        # FPS
        now = time.time()
        dt = now - prev_time
        if dt > 0:
            fps = 1.0 / dt
        prev_time = now

        # Draw overlays
        if DRAW:
            cur_disp = committed_char if committed_char is not None else top_label
            cur_conf = avg_conf if committed_char is None else 1.0
            draw_overlay(frame, committed_text, cur_disp, cur_conf, bbox=bbox, fps=fps)

        cv2.imshow("ASL Gesture Typing", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            committed_text = ""
        elif key == ord('b'):
            committed_text = committed_text[:-1]

    cap.release()
    cv2.destroyAllWindows()
    print("\nFinal text:", committed_text)


if __name__ == "__main__":
    main()
