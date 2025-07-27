import torch
from torch.nn import CTCLoss
from torch.utils.data import DataLoader
from model import ShorthandModel
from dataset import ShorthandDataset
from utils import get_char_map

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load character map and dataset
char_map = get_char_map()
dataset = ShorthandDataset("labels/labels.csv", "augmented", char_map)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Model setup
model = ShorthandModel(num_classes=len(char_map) + 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = CTCLoss(blank=0)

# Training loop
for epoch in range(35):
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        logits = model(imgs)  # [B, W, C]
        log_probs = logits.log_softmax(2).permute(1, 0, 2)  # [W, B, C]

        input_lengths = torch.full(size=(imgs.size(0),), fill_value=logits.size(1), dtype=torch.long)
        target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

        # CTC expects flattened labels
        targets = torch.cat([label for label in labels])

        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model.pth")
