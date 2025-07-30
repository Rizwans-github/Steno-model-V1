import torch
from model import ShorthandModel
from utils import get_char_map
from PIL import Image
import torchvision.transforms as T

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
char_map = get_char_map()
idx_map = {v: k for k, v in char_map.items()}  # reverse map

# Model
model = ShorthandModel(num_classes=len(char_map) + 1)
model.load_state_dict(torch.load("model.pth", map_location=device))  # <-- We'll save model after this
model.eval()

# Image
img = Image.open("augmented/ng_0.png").convert("RGB")
transform = T.Compose([
    T.Grayscale(),
    T.Resize((32, 128)),
    T.ToTensor()
])
img = transform(img).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(img)  # [B, W, C]
    probs = output.softmax(2)
    pred = probs.argmax(2)[0]  # first batch item
    print("Raw indices:", pred.tolist())
    pred_text = []
    prev = 0
    for p in pred:
        p = p.item()
        if p != prev and p != 0:
            pred_text.append(idx_map.get(p, "?"))
        prev = p

print("Prediction:", "".join(pred_text))
