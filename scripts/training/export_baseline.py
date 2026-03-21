"""Export pretrained Faster R-CNN to ONNX for baseline submission."""
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from pathlib import Path

ROOT = Path(__file__).parent

print("Loading pretrained Faster R-CNN...")
model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
model.eval()

# Save state dict (smaller, compatible)
out = ROOT / "weights"
out.mkdir(exist_ok=True)
torch.save(model.state_dict(), out / "fasterrcnn_coco.pt")
print(f"Saved: {(out / 'fasterrcnn_coco.pt').stat().st_size / 1e6:.1f} MB")

# Quick test
dummy = torch.randn(1, 3, 800, 600)
with torch.no_grad():
    outputs = model([dummy[0]])
print(f"Test output: {len(outputs[0]['boxes'])} detections")
print(f"Labels: {outputs[0]['labels'][:10].tolist()}")
print(f"Scores: {outputs[0]['scores'][:10].tolist()}")
