import argparse
import os
from pathlib import Path
from typing import List, Optional

import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import squeezenet1_0, SqueezeNet1_0_Weights, efficientnet_v2_s
from PIL import Image

# -----------------------------
# Custom SmallCNN definition
# -----------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# -----------------------------
# Helper functions
# -----------------------------
def load_class_names(classes_path: Optional[str]) -> Optional[List[str]]:
    if classes_path is None:
        return None
    p = Path(classes_path)
    if not p.exists():
        raise FileNotFoundError(f"Classes file not found: {classes_path}")
    with p.open('r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return lines

def infer_num_classes_from_weights(weights_path: str) -> int:
    sd = torch.load(weights_path, map_location='cpu')
    for k, v in sd.items():
        if v.ndim == 2:
            return v.shape[0]
    raise RuntimeError("Could not infer num_classes from state_dict; please provide classes file.")

def build_preprocess(model_name: str):
    # EfficientNet expects normalization
    if model_name in ["smallcnn", "squeezenet"]:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    else:  # efficientnet
        return transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])


# -----------------------------
# Model builder
# -----------------------------
def build_model(model_name: str, num_classes: int, device: torch.device):
    if model_name == "smallcnn":
        model = SmallCNN(num_classes=num_classes)
    elif model_name == "squeezenet":
        weights = SqueezeNet1_0_Weights.IMAGENET1K_V1
        model = squeezenet1_0(weights=weights)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1))
        model.num_classes = num_classes
    elif model_name == "efficientnet":
        model = efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Invalid model_name. Choose from smallcnn, squeezenet, efficientnet.")
    return model.to(device)

# -----------------------------
# Main inference routine
# -----------------------------
def infer_video(video_path: str, model_name: str, weights: str,
                classes: Optional[List[str]] = None, output_path: Optional[str] = None,
                device: Optional[torch.device] = None, show: bool = True):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Weights not found: {weights}")

    if classes is None:
        num_classes = infer_num_classes_from_weights(weights)
        class_names = [str(i) for i in range(num_classes)]
    else:
        class_names = classes
        num_classes = len(class_names)

    # Load model and weights
    model = build_model(model_name, num_classes, device)
    state_dict = torch.load(weights, map_location=device)
    new_state = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '')  # remove DataParallel prefix
        new_state[new_k] = v
    model.load_state_dict(new_state)
    model.eval()

    preprocess = build_preprocess(model_name)

    # Video setup
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, input_fps, (orig_w, orig_h))

    frame_idx = 0
    counts = {name: 0 for name in class_names}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            input_tensor = preprocess(pil).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(probs.argmax())
                confidence = float(probs[pred_idx])

            pred_label = class_names[pred_idx]
            counts[pred_label] += 1

            # Overlay text
            text = f"{pred_label} ({confidence*100:.1f}%)"
            (text_size, baseline) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            tx, ty = text_size
            rect_w = tx + 12
            rect_h = ty + 12
            x, y = 10, 30
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y-20), (x+rect_w, y-20+rect_h), (0,0,0), -1)
            alpha = 0.5
            frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
            cv2.putText(frame, text, (x+6, y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if show:
                cv2.imshow("Inference", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if writer:
                writer.write(frame)

            frame_idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    print("\nInference finished.")
    print("Predicted counts per class:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a video with multiple models.")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="smallcnn",
                        choices=["smallcnn", "squeezenet", "efficientnet"])
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--classes", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--no_display", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    classes_list = None
    if args.classes:
        classes_list = load_class_names(args.classes)

    infer_video(
        video_path=args.video_path,
        model_name=args.model,
        weights=args.weights,
        classes=classes_list,
        output_path=args.output_path,
        show=not args.no_display
    )
