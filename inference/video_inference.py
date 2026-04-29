import sys
from pathlib import Path
import torch
import cv2
from PIL import Image

sys.path.append(str(Path(__file__).parent))

from infer_smallcnn_video import build_model, build_preprocess, load_class_names

class VideoInferenceWrapper:
    def __init__(self, model_type="squeezenet", weights_path=None, classes_path="inference/classes.txt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        
        self.classes = load_class_names(classes_path) if classes_path else None
        self.num_classes = len(self.classes) if self.classes else 6
        
        self.model = build_model(self.model_type, self.num_classes, self.device)
        
        if weights_path and Path(weights_path).exists():
            state_dict = torch.load(weights_path, map_location=self.device)
            new_state = {}
            for k, v in state_dict.items():
                new_k = k.replace('module.', '')
                new_state[new_k] = v
            self.model.load_state_dict(new_state)
        
        self.model.eval()
        self.preprocess = build_preprocess(self.model_type)
    
    def get_model_name(self):
        return self.model_type
    
    def get_classes(self):
        return self.classes

    def predict(self, frame):
        """Predict class label for a single OpenCV frame (BGR ndarray)."""
        if frame is None:
            raise ValueError("Input frame cannot be None")

        if self.classes is None:
            raise RuntimeError("Class names are not loaded; provide a valid classes_path")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        input_tensor = self.preprocess(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            pred_idx = int(torch.argmax(logits, dim=1).item())

        return self.classes[pred_idx]