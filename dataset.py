import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as T

class ShorthandDataset(Dataset):
    def __init__(self, csv_file, img_dir, char_map):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.char_map = char_map
        self.transform = T.Compose([
            T.Grayscale(),
            T.Resize((32, 128)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = f"{self.img_dir}/{row['filename']}"
        img = Image.open(img_path).convert("RGB")
        label_text = row['text'].lower()
        label = [self.char_map[c] for c in label_text if c in self.char_map]
        return self.transform(img), torch.tensor(label, dtype=torch.long)
