import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import EmotionDataset
from model import CNN_LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

CSV_PATH = "features/data.csv"
MODEL_PATH = "checkpoints/cnn_lstm_ravdess.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
dataset = EmotionDataset(CSV_PATH)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Labels
emotion_classes = sorted(dataset.label_map.keys())

# Load model
model = CNN_LSTM(num_classes=len(emotion_classes)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Accuracy
accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
print(f"\nOverall Accuracy: {accuracy * 100:.2f}%\n")

# Classification Report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=emotion_classes))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=emotion_classes,
            yticklabels=emotion_classes,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Audio Emotion Recognition")
plt.savefig("outputs/confusion_matrix.png")
plt.show()

print("\nConfusion matrix saved to outputs/confusion_matrix.png")
