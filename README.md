:-> Cats vs Dogs Classifier

This project is a simple image classification web application that predicts whether an uploaded image is a cat or a dog.

I built this project to understand how pretrained deep learning models can be used in real-world applications. It helped me learn how machine learning models, APIs, and web interfaces work together.

-> Features

- Upload an image from your device
- Predict whether the image is a cat or a dog
- Display the prediction confidence
- Simple web interface for user interaction

-> Technologies Used

- Python
- PyTorch
- FastAPI
- HTML, CSS and JavaScript
- ResNet18 (Transfer Learning)
- Kaggle Cats vs Dogs Dataset

-> Project Structure

- `model.py` – Defines the ResNet18 model used for classification.
- `dataset.py` – Loads and preprocesses the training images.
- `train.py` – Fine-tunes the pretrained model using the Cats vs Dogs dataset.
- `predict.py` – Generates predictions for new images.
- `app.py` – FastAPI backend that handles prediction requests.
- `index.html` – Frontend interface for uploading images and displaying results.


-> How It Works

The user uploads an image through the web page. The image is sent to the FastAPI backend, where the trained model processes it and returns the prediction along with its confidence score. The result is then displayed on the screen.

-> What I Learned

- Working with pretrained models using transfer learning
- Training and testing image classification models
- Building APIs using FastAPI
- Connecting frontend and backend components
- Understanding the complete workflow of a small AI application

-> Future Improvements

- Support more animal categories
- Improve the user interface
- Deploy the application online
- Experiment with different models for better performance

## Note

This project was built as a learning project to gain hands-on experience in deep learning and full-stack integration.
