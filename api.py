import io
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import torch.nn as nn

# ----------- 1. Init basic structure of the AI -----------

# define base construct for ai
model = models.resnet18()

# Reducing the classificator-"head" to my 2 classes (cat and dog)
# The weights will be defined by the predefined .pth file from the pretraining and testing process
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Load in the .pth file as the weights
model.load_state_dict(torch.load('cat_dog_cnn.pth'))

# Set GPU if existing (if not use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # Because I wont train with the API I want to evaluate

# Changing the input image size to 160x160
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# Define the 2 classes "cats" and "dogs"
class_names = ['cats', 'dogs']

# ----------- 2. The real shit -> The API Stuff -----------

# Creating a Fast-API instance
app = FastAPI(title="Cats & Dogs Classificator API")

@app.get("/")
def read_root():
    """ Prints out a initialization message. """
    return {"message": "FastAPI initialized successfully. Go to /docs to test the API."}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Takes the input as image and predicts if the image represents a Dog or a Cat.
    """
    # load image (to make it useable)
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # Make the image compatible for the model
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict if the image represents a Cat or a Dog (magic stuff lol)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_index = torch.max(outputs, 1)

    prediction = class_names[predicted_index.item()]

    # Return as JSON file
    return {
        "filename": file.filename,
        "prediction": prediction
    }