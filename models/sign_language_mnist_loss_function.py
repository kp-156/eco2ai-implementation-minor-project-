import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SignDataSet(Dataset):
    def __init__(
            self,
            image_df,
            label_df,
            transform,
            split=None,
    ):
        self.image_df = image_df
        self.label_df = torch.nn.functional.one_hot(torch.tensor(np.array(label_df))).float()
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, index):
        image = self.image_df.iloc[index]
        image = np.reshape(np.array(image), (28, 28))

        image = Image.fromarray(image.astype(np.uint8))

        label = self.label_df[index]
        # label = torch.nn.functional.one_hot(torch.tensor(label))

        if self.split == 'train':
            image = self.transform(image)
        return image, label


class SignLabelModel(nn.Module):
    def __init__(self, num_classes):
        super(SignLabelModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_model():
    """
    Train the model
    :return:
    """
    # Load the datasets into pandas dataframe
    train = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/sign_mnist_train.csv'))
    test = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/sign_mnist_test.csv'))

    x = train.drop(['label'], axis=1)
    y = train['label']
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    print(f"Training and testing datasets have been loaded. Train dataset: {x.shape}, Test dataset: {y.shape}")

    # Define Hyper-parameters
    BATCH_SIZE = 32
    IMAGE_SIZE = 28
    NUM_CLASS = y.nunique() + 1

    # Data augmentation
    random_transforms = transforms.Compose([
        transforms.RandomRotation(30),  # Randomly rotate the image by up to 30 degrees
        # transforms.RandomResizedCrop(IMAGE_SIZE),  # Randomly crop and resize the image to 224x224
        # transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    ])

    # Define the fixed transformations
    fixed_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Define the overall transformation pipeline
    transform = transforms.Compose([
        transforms.RandomApply([random_transforms], p=0.5),  # Apply random transformations with a probability of 0.5
        fixed_transforms
    ])
    print("Data preprocessing and transformation is completed.")

    train_data = SignDataSet(x_train, y_train, transform, 'train')
    test_data = SignDataSet(x_val, y_val, transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, drop_last=True)
    print("Sign language dataset has been loaded into the dataloader.")

    # Create an instance of the model with number of classes
    model = SignLabelModel(NUM_CLASS).to(device)

    # Define the loss function and optimizer
    num_epochs = 20
    criterion = nn.HingeEmbeddingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    print("Model training started...")
    for epoch in tqdm(range(num_epochs)):
        for i, (images, labels) in enumerate(train_loader):
            # Move images and labels to the device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images.to(device))
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print training progress
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Training finished")
