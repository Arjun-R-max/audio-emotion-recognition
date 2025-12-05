import torch
from model import CNN_LSTM
from torchvision import transforms
from PIL import Image

emotions = ["neutral","calm","happy","sad","angry","fear","disgust","surprise"]

model = CNN_LSTM(len(emotions))
model.load_state_dict(torch.load("checkpoints/cnn_lstm_ravdess.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

def predict(mel_image_path):
    img = Image.open(mel_image_path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)

    outputs = model(img)
    _, pred = torch.max(outputs, 1)
    return emotions[pred.item()]
print(predict("features/mel_specs/03-01-06-01-02-02-01.png"))
