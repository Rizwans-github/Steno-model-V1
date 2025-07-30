from dataset import ShorthandDataset
from utils import get_char_map
import matplotlib.pyplot as plt
import torch

char_map = get_char_map()
dataset = ShorthandDataset("C:/Users/rizwa/Tech/Github/Steno/labels/labels.csv", 
                           "C:/Users/rizwa/Tech/Github/Steno/augmented", 
                           char_map)
print("Total samples:", len(dataset))

# Load 1 sample
img, label = dataset[0]

print("Image shape:", img.shape)
print("Label (as IDs):", label.tolist())

# Show the image (optional)
plt.imshow(img.squeeze(0), cmap="gray")
plt.title("Shorthand Sample")
plt.axis("off")
plt.show()
