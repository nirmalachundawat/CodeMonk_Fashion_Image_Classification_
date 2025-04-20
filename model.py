import pickle
import os
import pandas as pd
# Load label encoders
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Set paths (update to your actual location)
dataset_path = '/kaggle/input/fashion-product-images-dataset/fashion-dataset'
styles_csv_path = os.path.join(dataset_path, 'styles.csv')

# Load CSV
df = pd.read_csv(styles_csv_path, on_bad_lines='skip')  # skip corrupt lines if any
df.head()
df.head()  # Uncomment if you want to see the top rows

# Basic info
print(df.info())

# Check missing values
print(df.isnull().sum())

# Unique classes for target columns
print("Unique Genders:", df['gender'].unique())
print("Unique Article Types:", df['articleType'].unique())
print("Unique Colors:", df['baseColour'].unique())
print("Unique Seasons:", df['season'].unique())

import matplotlib.pyplot as plt
from PIL import Image

# Add image path to dataframe
df['image_path'] = df['id'].astype(str) + ".jpg"

# Drop rows where the image file is missing
df['full_path'] = df['image_path'].apply(lambda x: os.path.join(dataset_path, 'images', x))
df = df[df['full_path'].apply(os.path.exists)]

# Plot some images
def show_samples(df_subset, n=5):
    plt.figure(figsize=(15, 6))
    for i in range(n):
        sample = df_subset.iloc[i]
        img = Image.open(sample['full_path'])
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.title(f"{sample['articleType']}\n{sample['baseColour']}, {sample['gender']}, {sample['season']}")
        plt.axis('off')
    plt.show()

# Display 5 samples
show_samples(df)

# Check basic info
print(df.info())

# Count missing values
print("\nMissing values:\n", df.isnull().sum())

# Drop rows with missing target labels (we need these to train)
df = df.dropna(subset=['gender', 'baseColour', 'season', 'articleType'])

print("Shape after dropping rows with missing labels:", df.shape)

import seaborn as sns
import matplotlib.pyplot as plt

# Gender Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='gender', order=df['gender'].value_counts().index)
plt.title('Gender Distribution')
plt.show()

# Base Colour Distribution (Top 10)
plt.figure(figsize=(10,4))
top_colors = df['baseColour'].value_counts().nlargest(10).index
sns.countplot(data=df[df['baseColour'].isin(top_colors)], x='baseColour', order=top_colors)
plt.title('Top 10 Product Colors')
plt.xticks(rotation=45)
plt.show()

# Season Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='season', order=df['season'].value_counts().index)
plt.title('Season Distribution')
plt.show()

# Article Type Distribution (Top 10)
plt.figure(figsize=(12,5))
top_types = df['articleType'].value_counts().nlargest(10).index
sns.countplot(data=df[df['articleType'].isin(top_types)], x='articleType', order=top_types)
plt.title('Top 10 Product Types')
plt.xticks(rotation=45)
plt.show()

from sklearn.preprocessing import LabelEncoder

# Encode and store encoders for each target
label_encoders = {}
target_columns = ['baseColour', 'articleType', 'season', 'gender']

for col in target_columns:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

# Example mapping
for col in target_columns:
    print(f"{col} classes:", list(label_encoders[col].classes_))

# Filter out articleTypes with fewer than 5 samples
min_count = 5
article_counts = df['articleType_encoded'].value_counts()
valid_articles = article_counts[article_counts >= min_count].index
df = df[df['articleType_encoded'].isin(valid_articles)]

print("Filtered dataset shape:", df.shape)

from sklearn.model_selection import train_test_split

df_train, df_temp = train_test_split(df, test_size=0.3, random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

print("Train size:", len(df_train))
print("Validation size:", len(df_val))
print("Test size:", len(df_test))

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


image_transforms = transforms.Compose([
    transforms.Resize((72, 72)),
    transforms.RandomCrop((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class FashionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['full_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        labels = {
            'baseColour': row['baseColour_encoded'],
            'articleType': row['articleType_encoded'],
            'season': row['season_encoded'],
            'gender': row['gender_encoded']
        }
        return image, labels

from torch.utils.data import DataLoader
df_train_small = df_train.sample(10000, random_state=42)
df_val_small = df_val.sample(2000, random_state=42)

train_dataset = FashionDataset(df_train_small, transform=image_transforms)
val_dataset = FashionDataset(df_val_small, transform=image_transforms)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

import torchvision.models as models

import torch.nn as nn

class ResNetMultiOutput(nn.Module):
    def __init__(self, num_colors, num_types, num_seasons, num_genders):
        super(ResNetMultiOutput, self).__init__()

        # Load pretrained resnet
        self.backbone = models.resnet18(pretrained=True)

        # Remove original fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Output heads
        self.fc_color = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_colors)
        )
        self.fc_type = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_types)
        )
        self.fc_season = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_seasons)
        )
        self.fc_gender = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_genders)
        )

    def forward(self, x):
        features = self.backbone(x)
        return {
            'baseColour': self.fc_color(features),
            'articleType': self.fc_type(features),
            'season': self.fc_season(features),
            'gender': self.fc_gender(features)
        }

if __name__ == "__main__":
    dataset_path = '/kaggle/input/fashion-product-images-dataset/fashion-dataset'
    styles_csv_path = os.path.join(dataset_path, 'styles.csv')

    df = pd.read_csv(styles_csv_path, on_bad_lines='skip')
    print("Dataset shape:", df.shape)
    print(df.head())

model = ResNetMultiOutput(
    num_colors=len(label_encoders['baseColour'].classes_),
    num_types=len(label_encoders['articleType'].classes_),
    num_seasons=len(label_encoders['season'].classes_),
    num_genders=len(label_encoders['gender'].classes_)
)


print(model)

import torch
import torch.optim as optim

# Loss functions for each output
criterion_color = nn.CrossEntropyLoss()
criterion_type = nn.CrossEntropyLoss()
criterion_season = nn.CrossEntropyLoss()
criterion_gender = nn.CrossEntropyLoss()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, val_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            optimizer.zero_grad()
            outputs = model(images)

            loss_color = criterion_color(outputs['baseColour'], labels['baseColour'])
            loss_type = criterion_type(outputs['articleType'], labels['articleType'])
            loss_season = criterion_season(outputs['season'], labels['season'])
            loss_gender = criterion_gender(outputs['gender'], labels['gender'])

            total_loss = loss_color + loss_type + loss_season + loss_gender
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}
                outputs = model(images)

                loss_color = criterion_color(outputs['baseColour'], labels['baseColour'])
                loss_type = criterion_type(outputs['articleType'], labels['articleType'])
                loss_season = criterion_season(outputs['season'], labels['season'])
                loss_gender = criterion_gender(outputs['gender'], labels['gender'])

                total_loss = loss_color + loss_type + loss_season + loss_gender
                val_loss += total_loss.item()
        model.train()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}")

train_model(model, train_loader, val_loader, epochs=10)

# Create test dataset and loader
test_dataset = FashionDataset(df_test, transform=image_transforms)
test_loader = DataLoader(test_dataset, batch_size=64)

# Set model to eval mode
model.eval()

test_loss = 0.0
correct = {'baseColour': 0, 'articleType': 0, 'season': 0, 'gender': 0}
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        outputs = model(images)

        loss_color = criterion_color(outputs['baseColour'], labels['baseColour'])
        loss_type = criterion_type(outputs['articleType'], labels['articleType'])
        loss_season = criterion_season(outputs['season'], labels['season'])
        loss_gender = criterion_gender(outputs['gender'], labels['gender'])

        total_loss = loss_color + loss_type + loss_season + loss_gender
        test_loss += total_loss.item()

        # Accuracy
        _, pred_color = torch.max(outputs['baseColour'], 1)
        _, pred_type = torch.max(outputs['articleType'], 1)
        _, pred_season = torch.max(outputs['season'], 1)
        _, pred_gender = torch.max(outputs['gender'], 1)

        correct['baseColour'] += (pred_color == labels['baseColour']).sum().item()
        correct['articleType'] += (pred_type == labels['articleType']).sum().item()
        correct['season'] += (pred_season == labels['season']).sum().item()
        correct['gender'] += (pred_gender == labels['gender']).sum().item()
        total += labels['baseColour'].size(0)

# Print final metrics
print(f"Test Loss: {test_loss / len(test_loader):.4f}")
print(f"Test Accuracy - baseColour: {correct['baseColour'] / total:.4f}")
print(f"Test Accuracy - articleType: {correct['articleType'] / total:.4f}")
print(f"Test Accuracy - season: {correct['season'] / total:.4f}")
print(f"Test Accuracy - gender: {correct['gender'] / total:.4f}")

model.train()  # Set back to train mode if needed

import random

# Switch to evaluation mode
model.eval()

def visualize_predictions(model, dataset, label_encoders, num_images=5):
    indices = random.sample(range(len(dataset)), num_images)
    plt.figure(figsize=(15, 6))

    for i, idx in enumerate(indices):
        image, true_labels = dataset[idx]
        input_img = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_img)

        preds = {key: torch.argmax(value, dim=1).item() for key, value in output.items()}

        decoded_preds = {
            key: label_encoders[key].inverse_transform([preds[key]])[0] for key in preds
        }
        decoded_trues = {
            key: label_encoders[key].inverse_transform([true_labels[key]])[0] for key in true_labels
        }

        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 0.5) + 0.5  # Unnormalize

        plt.subplot(1, num_images, i + 1)
        plt.imshow(img_np)
        plt.axis('off')
        plt.title(
            f"Pred:\n{decoded_preds['articleType']}, {decoded_preds['baseColour']}\n"
            f"{decoded_preds['gender']}, {decoded_preds['season']}\n\n"
            f"True:\n{decoded_trues['articleType']}, {decoded_trues['baseColour']}\n"
            f"{decoded_trues['gender']}, {decoded_trues['season']}",
            fontsize=8
        )

    plt.tight_layout()
    plt.show()

# Visualize predictions
visualize_predictions(model, test_dataset, label_encoders, num_images=5)

import os

input_dir = '/kaggle/input'  # Default location for Kaggle input datasets
print("Input directory contents:", os.listdir(input_dir))

# test_images_dir = '/kaggle/input/test-sample-dataset'  # This is where your test images are
# print("Test images:", os.listdir(test_images_dir))

from sklearn.preprocessing import LabelEncoder

# Re-create the label encoders
label_encoders = {}
target_columns = ['baseColour', 'articleType', 'season', 'gender']

for col in target_columns:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save each encoder for easier access
le_colour = label_encoders['baseColour']
le_articleType = label_encoders['articleType']
le_season = label_encoders['season']
le_gender = label_encoders['gender']

import os
from PIL import Image
import torch
from torchvision import transforms

# ✅ Updated path for Colab
image_path = "/content/cap.jpg"  # Image you uploaded in Colab

# Image transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load and transform the image
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0)  # Add batch dimension

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = image.to(device)

# Move model to device
model.to(device)
model.eval()

# Inference
with torch.no_grad():
    outputs = model(image)

# Get predicted indices
predicted_color = torch.argmax(outputs['baseColour'], dim=1).item()
predicted_type = torch.argmax(outputs['articleType'], dim=1).item()
predicted_season = torch.argmax(outputs['season'], dim=1).item()
predicted_gender = torch.argmax(outputs['gender'], dim=1).item()

# Decode predictions
color_label = label_encoders['baseColour'].classes_[predicted_color]
type_label = label_encoders['articleType'].classes_[predicted_type]
season_label = label_encoders['season'].classes_[predicted_season]
gender_label = label_encoders['gender'].classes_[predicted_gender]

# Output results
print(f"🎨 Predicted Color: {color_label}")
print(f"👕 Predicted Article Type: {type_label}")
print(f"🍁 Predicted Season: {season_label}")
print(f"🚻 Predicted Gender: {gender_label}")

from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

all_preds = {'baseColour': [], 'articleType': [], 'season': [], 'gender': []}
all_labels = {'baseColour': [], 'articleType': [], 'season': [], 'gender': []}

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        outputs = model(images)

        for key in all_preds.keys():
            preds = torch.argmax(outputs[key], dim=1)
            all_preds[key].extend(preds.cpu().numpy())
            all_labels[key].extend(labels[key].cpu().numpy())

# Plot confusion matrices and accuracy
for key in all_preds:
    cm = confusion_matrix(all_labels[key], all_preds[key])
    acc = accuracy_score(all_labels[key], all_preds[key])

    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {key} (Accuracy: {acc:.2f})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Save only the model weights (recommended)
torch.save(model.state_dict(), "fashion_model.pth")

import pickle

# Assuming these are your fitted label encoders:
# le_colour, le_articleType, le_season, le_gender

label_encoders = {
    'baseColour': le_colour,
    'articleType': le_articleType,
    'season': le_season,
    'gender': le_gender
}

# Save to a pickle file
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
print("✅ label_encoders.pkl saved successfully.")













import os
from PIL import Image
import torch
from torchvision import transforms

# Path to the test image directory
test_images_dir = "/kaggle/input/test-sample-dataset"  # Update if different
image_filename = "skirt.jpg"  # Replace with your test image
image_path = os.path.join(test_images_dir, image_filename)

# Load and transform the image
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0)  # Add batch dimension

# Move image to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = image.to(device)

# Ensure model is on the correct device
model.to(device)
model.eval()

# Make prediction
with torch.no_grad():
    outputs = model(image)

# Get predicted class indices
predicted_color = torch.argmax(outputs['baseColour'], dim=1).item()
predicted_type = torch.argmax(outputs['articleType'], dim=1).item()
predicted_season = torch.argmax(outputs['season'], dim=1).item()
predicted_gender = torch.argmax(outputs['gender'], dim=1).item()

# Decode predictions to labels
color_label = label_encoders['baseColour'].classes_[predicted_color]
type_label = label_encoders['articleType'].classes_[predicted_type]
season_label = label_encoders['season'].classes_[predicted_season]
gender_label = label_encoders['gender'].classes_[predicted_gender]

# Output
print(f"Predicted Color: {color_label}")
print(f"Predicted Article Type: {type_label}")
print(f"Predicted Season: {season_label}")
print(f"Predicted Gender: {gender_label}")

