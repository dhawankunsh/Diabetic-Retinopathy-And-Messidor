import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from skimage.filters import frangi  # Vessel enhancement

# Paths to dataset and CSV
IMAGE_PATH = "messidor-2/preprocess"
CSV_PATH = "messidor_data.csv"

def preprocess_image(img):
    img = cv2.resize(img, (512, 512))  # Increased resolution for more details
    img = img[:, :, 1]  # Extract green channel (better contrast for retinas)
    
    img = cv2.medianBlur(img, 5)  # Use Median Blur instead of Gaussian
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Apply Frangi filter for vessel enhancement
    vessel_img = frangi(img)
    vessel_img = (vessel_img * 255).astype(np.uint8)  # Normalize to 0-255

    return np.hstack((img.flatten(), vessel_img.flatten()))

def load_images_labels(image_path, csv_path):
    df = pd.read_csv(csv_path)
    images, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading Images"):
        img_name, label = row["id_code"], row["diagnosis"]
        img_path = os.path.join(image_path, img_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = preprocess_image(img)
            images.append(img)
            labels.append(int(label))
    return np.array(images), np.array(labels)

# Load dataset
X, y = load_images_labels(IMAGE_PATH, CSV_PATH)

# Apply LDA for dimensionality reduction
lda = LDA(n_components=min(len(np.unique(y)) - 1, 100))  # LDA needs classes-1 components
X = lda.fit_transform(X, y)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Use Stratified Shuffle Split for better class distribution
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train_full, X_test = X[train_idx], X[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]

# Further split training data into 80% training and 20% validation
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in sss.split(X_train_full, y_train_full):
    X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
    y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

# Train Naive Bayes model
print("Training model...")
model = GaussianNB(var_smoothing=1e-7)
model.fit(X_train, y_train)
print("Model training complete.")

# Validate model
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Test model
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

print("Program by Adit Vij 0221BCA042")
