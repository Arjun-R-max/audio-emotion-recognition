import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

AUDIO_DIR = "data/ravdess/"
OUTPUT_DIR = "features/mel_specs/"
CSV_PATH = "features/data.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprise'
}

rows = []

for root, dirs, files in os.walk(AUDIO_DIR):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)

            emotion_code = file.split("-")[2]
            emotion = emotion_map.get(emotion_code, "unknown")

            y, sr = librosa.load(file_path, sr=22050)
            y = librosa.util.normalize(y)

            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel = librosa.power_to_db(mel, ref=np.max)

            img_file = file.replace(".wav", ".png")
            img_path = os.path.join(OUTPUT_DIR, img_file)

            plt.figure(figsize=(2, 2))
            librosa.display.specshow(mel, sr=sr)
            plt.axis("off")
            plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            rows.append([img_path, emotion])

df = pd.DataFrame(rows, columns=["filepath", "emotion"])
df.to_csv(CSV_PATH, index=False)

print("Feature extraction complete!")
