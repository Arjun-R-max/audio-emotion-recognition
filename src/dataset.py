import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms

class EmotionDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        self.emotions = sorted(self.data["emotion"].unique())
        self.label_map = {e:i for i,e in enumerate(self.emotions)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["filepath"]
        label = self.label_map[self.data.iloc[idx]["emotion"]]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label
