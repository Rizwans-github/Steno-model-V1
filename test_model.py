import torch
from model import ShorthandModel
from utils import get_char_map

char_map = get_char_map()
model = ShorthandModel(num_classes=len(char_map) + 1)  # +1 for CTC blank

dummy_input = torch.randn(1, 1, 32, 128)  # [batch, channels, height, width]
output = model(dummy_input)

print("Output shape:", output.shape)  # Expect: [1, width, num_classes]

