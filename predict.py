import torch
from torchvision import transforms
from PIL import Image
from model import get_model

def predict(image_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = get_model().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

    classes = ["Cat", "Dog"]
    return {
        "prediction": classes[predicted.item()],
        "confidence": f"{confidence.item() * 100:.2f}%"
    }

if __name__ == "__main__":
    result = predict("test_image.jpg")
    print(result)