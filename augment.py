from PIL import Image
import torchvision.transforms as T
import os

input_dir = "images"
output_dir = "augmented"
os.makedirs(output_dir, exist_ok=True)

augmentations = T.Compose([
    T.RandomRotation(degrees=10),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),         # Convert to tensor for transforms
    T.ToPILImage()        # Convert back to image for saving
])

# How many versions to make per image
copies = 10

for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        base_img = Image.open(os.path.join(input_dir, filename)).convert("RGB")

        for i in range(copies):
            aug_img = augmentations(base_img)
            out_name = f"{filename[:-4]}_{i}.png"
            aug_img.save(os.path.join(output_dir, out_name))

print(f"✅ Augmented {copies}x images saved to /{output_dir}")
