import torch
from torch.utils.data import DataLoader
from dataset import EmotionDataset
from model import CNN_LSTM
import torch.nn as nn
import torch.optim as optim

dataset = EmotionDataset("features/data.csv")
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN_LSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 25

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}: Loss = {total_loss:.4f}")

torch.save(model.state_dict(), "checkpoints/cnn_lstm_ravdess.pth")
print("Training complete!")
