 Cats vs Dogs Image Classifier

An AI-powered web application that classifies uploaded images as either a cat or a dog using Deep Learning and Transfer Learning techniques.

 📌 Project Overview

This project uses a pretrained ResNet18 model to identify whether an uploaded image contains a cat or a dog. The model is fine-tuned using the Cats vs Dogs dataset and deployed through a FastAPI backend with a simple web interface.

->Features

- Upload an image through a web interface
- Predict whether the image is a cat or a dog
- Display prediction confidence score
- FastAPI backend for handling requests
- Deep learning model built using PyTorch
- Uses Transfer Learning with ResNet18

-> 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- FastAPI
- Uvicorn
- Pillow
- HTML
- CSS
- JavaScript
- Kaggle Cats vs Dogs Dataset

-> Project Structure

```
project/
│
├── model.py
├── dataset.py
├── train.py
├── predict.py
├── app.py
├── index.html
├── requirements.txt
└── README.md
```
-> 🧠 How It Works

1. The user uploads an image through the frontend.
2. The image is sent to the FastAPI backend.
3. The trained ResNet18 model processes the image.
4. The model predicts whether it is a cat or a dog.
5. The result and confidence score are displayed to the user.

-> Running the Project

--> Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

-> Install dependencies

```bash
pip install -r requirements.txt
```

--> Start the FastAPI server

```bash
uvicorn app:app --reload
```

-->Open in browser

Visit:

```
http://127.0.0.1:8000
```

-> Model Information

- Model: ResNet18
- Technique: Transfer Learning
- Task: Binary Image Classification (Cat vs Dog)

-> Learning Outcomes

Through this project, I gained practical experience in:

- Deep Learning with PyTorch
- Transfer Learning
- Building REST APIs using FastAPI
- Frontend and Backend integration
- Image preprocessing and prediction workflows

## 📄 License

This project is created for educational and learning purposes.
