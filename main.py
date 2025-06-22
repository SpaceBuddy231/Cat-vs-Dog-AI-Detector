import os
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

cwd = os.getcwd()

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer: 3 input channels (RGB), 16 output channels
        # Kernel size 3x3, padding=1 to preserve spatial dimensions
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # Second convolutional layer: 16 input channels, 32 output channels
        # Kernel size 3x3, padding=1 to preserve spatial dimensions
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Max pooling layer, halves spatial dimensions (160x160 → 80x80 → 40x40)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers for classification
        # Input features: 32 * 40 * 40
        self.fc1 = nn.Linear(32 * 40 * 40, 512)
        # Output layer: 2 classes (cat, dog)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        # conv1 → ReLU → pool
        x = self.pool(F.relu(self.conv1(x)))
        # conv2 → ReLU → pool
        x = self.pool(F.relu(self.conv2(x)))
        # flatten feature maps
        x = x.view(-1, 32 * 40 * 40)
        # first fully connected layer with ReLU
        x = F.relu(self.fc1(x))
        # final output layer
        x = self.fc2(x)
        return x


def train_model(model, criterion, optimizer, train_loader, num_epochs):
    print("\n\n\n”<--- Starting Training --->\n")
    # Starting a for loop for the number of epochs (more eporchs = more training -> more accuracy)
    for epoch in range(num_epochs):
        # Loop through the training data
        print(f"Epoch {epoch+1}/{num_epochs}")
        for i, (images, labels) in enumerate(train_loader):
            # Step 1: AI guesses if the image is represents a cat or a dog
            outputs = model(images)

            # Step 2: 'loss' defines a 'how wrong' rate | This rate will be used to show the AI how ass it was in its guess
            loss = criterion(outputs, labels)

            # Step 3: The old corrections are deleted that the optimizer can learn from the new ones
            optimizer.zero_grad()

            # Step 4: The smart-ass torch system looks for the magical parameters that are responsible for the loss rate
            loss.backward()

            # Step 5: Improve the model (magic magic)
            optimizer.step()

            # Print information every 100 steps
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    print("\n<--- Finished Training --->\n")

def test_model(model, test_loader):
    print("--- Starting Testing ---")
    # This step appears to be important for Pytorch to tell the model that we are in evaluation mode
    model.eval()

    # Disable gradient calculation (this is needed in the training phase but not in the > evaluation < phase)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            # Let the AI decide if the image is a cat or a dog
            outputs = model(images)

            # Find the class with the highest probability
            # outputs.data gives us the raw numbers, torch.max finds the largest value in each row (dim=1)
            _, predicted = torch.max(outputs.data, 1)

            # Count the total number of images
            total += labels.size(0)

            # Count how many images were correctly guessed
            correct += (predicted == labels).sum().item()

    # Calculate and print the accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the {total} test images: {accuracy:.2f} %')


def init_data():
    data = cwd + '/data'
    if not os.path.exists(data):
        print("./data does not exist. Without the data folder program will not work. Aborting.")
        exit(1)

    train_dir = data + '/training_set'
    test_dir = data + '/test_set'

    if not os.path.exists(train_dir):
        print("Train data folders do not exist. Without the train data program will not work. Aborting.")
        exit(1)

    if not os.path.exists(test_dir):
        print("Test data folders do not exist. Without the test data program will not work. Aborting.")
        exit(1)

    BATCH_SIZE = 32
    IMG_SIZE = (160, 160)

    # Setting up the training dataset
    train_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Setting up the test dataset
    test_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Datasets created (train): {train_dataset.classes} and (test) {test_dataset.classes}")
    print(f"Creating Convolutional Neural Network (CNN) model...")

    model = SimpleCNN()
    print(model)

    # Defining an 'teacher' like optimizer
    criterion = nn.CrossEntropyLoss()

    # Giving the 'teacher' the 'teachable' parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("Model created.")

    train_model(model, criterion, optimizer, train_loader, num_epochs=10)

    test_model(model, test_loader)

    # Save the trained and tested model (because it would be a waste of time to train it again and again every single time)
    torch.save(model.state_dict(), 'cat_dog_cnn.pth')


init_data()
