"""DenseNet-121 CNN for AI-generated vs real image/video scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import densenet121

IMG_SIZE = 224
NUM_VIDEO_FRAMES = 8
HF_REPO = "VilaVision/AIgeneratedimagedetection"
HF_WEIGHTS = "densenet_finetuned_dense.pth"

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None
_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_model() -> torch.nn.Module:
    """Download weights on first call and cache the CNN in memory."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    weights_path = hf_hub_download(repo_id=HF_REPO, filename=HF_WEIGHTS)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 3)
    model.load_state_dict(state)
    model.to(_DEVICE).eval()
    _MODEL = model
    return _MODEL


def _scores_from_logits(logits: torch.Tensor) -> Dict[str, Union[str, float]]:
    probs = torch.nn.functional.softmax(logits, dim=1)[0]
    ai_prob = float((probs[0] + probs[2]).item())
    real_prob = float(probs[1].item())
    return {
        "label": "AI-generated" if ai_prob >= real_prob else "Real",
        "ai_probability": ai_prob,
        "real_probability": real_prob,
        "confidence": max(ai_prob, real_prob),
    }


@torch.no_grad()
def predict_pil(image: Image.Image) -> Dict[str, Union[str, float]]:
    model = load_model()
    rgb = image.convert("RGB")
    batch = _TRANSFORM(rgb).unsqueeze(0).to(_DEVICE)
    logits = model(batch)
    return _scores_from_logits(logits)


def predict_image(source: Union[str, Path, Image.Image]) -> Dict[str, Union[str, float]]:
    if isinstance(source, Image.Image):
        return predict_pil(source)
    return predict_pil(Image.open(source))


def _sample_frame_indices(frame_count: int, n: int) -> List[int]:
    if frame_count <= 0:
        return []
    n = min(n, frame_count)
    return [int(round(i * (frame_count - 1) / max(n - 1, 1))) for i in range(n)]


def predict_video(path: Union[str, Path], num_frames: int = NUM_VIDEO_FRAMES) -> Dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Could not open video file.")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = _sample_frame_indices(frame_count, num_frames)
    wanted = set(indices)
    frames: Dict[int, Image.Image] = {}

    if not wanted:
        ok, bgr = capture.read()
        if not ok:
            capture.release()
            raise ValueError("Video has no readable frames.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames[0] = Image.fromarray(rgb)
        indices = [0]
    else:
        idx = 0
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if idx in wanted:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames[idx] = Image.fromarray(rgb)
            idx += 1
            if len(frames) == len(wanted):
                break
    capture.release()

    if not frames:
        raise ValueError("Could not decode any video frames.")

    per_frame = []
    ai_scores = []
    for i, frame_index in enumerate(indices):
        image = frames.get(frame_index)
        if image is None:
            continue
        result = predict_pil(image)
        ai_scores.append(result["ai_probability"])
        per_frame.append(
            {
                "frame": i + 1,
                "frame_index": frame_index,
                **result,
            }
        )

    if not ai_scores:
        raise ValueError("Could not score any video frames.")

    ai_prob = float(np.mean(ai_scores))
    real_prob = 1.0 - ai_prob
    label = "AI-generated" if ai_prob >= real_prob else "Real"
    return {
        "label": label,
        "ai_probability": ai_prob,
        "real_probability": real_prob,
        "confidence": max(ai_prob, real_prob),
        "frames_scored": len(per_frame),
        "per_frame": per_frame,
    }
