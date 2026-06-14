from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import io
from model import get_model

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Load model once at startup
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
    return {
        "prediction": classes[predicted.item()],
        "confidence": round(confidence.item() * 100, 2),
        "emoji": "🐱" if predicted.item() == 0 else "🐶"
    }