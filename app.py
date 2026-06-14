from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import requests
from model import get_model

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Download model from Hugging Face
MODEL_URL = "https://huggingface.co/adithya36manoj/cats-vs-dogs-model/resolve/main/best_model.pth"

def download_model():
    if not os.path.exists("best_model.pth"):
        print("Downloading model...")
        r = requests.get(MODEL_URL)
        with open("best_model.pth", "wb") as f:
            f.write(r.content)
        print("Done!")

download_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

    classes = ["Cat", "Dog"]
    emojis = ["🐱", "🐶"]
    return {
        "prediction": classes[predicted.item()],
        "confidence": round(confidence.item() * 100, 2),
        "emoji": emojis[predicted.item()]
    }