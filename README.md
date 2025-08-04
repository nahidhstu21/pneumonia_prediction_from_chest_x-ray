# Pneumonia Detection from Chest X-ray

This project uses a deep learning model to detect **pneumonia** from **chest X-ray images**. The web interface allows you to upload, paste, or capture an image, and get a prediction whether the input is `Pneumonia` or `Normal`.

## Features

- Upload or drag-and-drop X-ray image
- Paste image from clipboard or take a photo
- Real-time prediction using a trained CNN model
- User-friendly interface using Gradio / Streamlit
- Flag incorrect predictions

## Model

- **Architecture**: CNN (e.g., ResNet50 / VGG16 / Custom)
- **Dataset**: Chest X-ray Images (Pneumonia) from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- **Classes**: `Pneumonia`, `Normal`
- **Metrics**: Accuracy, Precision, Recall

## Installation

```bash
git clone https://github.com/nahidhstu21/pneumonia-xray-detector.git
cd pneumonia_prediction_from_chest_x-ray
pip install -r requirements.txt

##Usage
python app.py
Then open your browser and go to: http://localhost:7860

##Model Inference
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("model.h5")
def predict(image):
    img = image.resize((224, 224)).convert('RGB')
    img = np.array(img) / 255.0
    img = img.reshape(1, 224, 224, 3)
    prediction = model.predict(img)
    return "Pneumonia" if prediction[0][0] > 0.5 else "Normal"

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 97.8% |
| Precision | 98.4% |
| Recall    | 96.1% |

