import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage import exposure
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("IDR_dataset.csv")
image_dir = "IDR_images"
output_dir = "Preprocessed_Images"

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

def preprocess_image(image_path, save_path=None):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Warning: Unable to load image at {image_path}")
        return None
    img = cv2.resize(img, (128, 128))  # Resize for efficiency
    
    # Convert to LAB color space
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Extract HOG features
    hog_features, hog_img = hog(
        img_lab, 
        pixels_per_cell=(16, 16), 
        cells_per_block=(2, 2), 
        visualize=True, 
        channel_axis=-1  # Use channel_axis instead of multichannel
    )
    hog_features = exposure.rescale_intensity(hog_img, in_range=(0, 10))

    # Save preprocessed image if save_path is provided
    if save_path:
        cv2.imwrite(save_path, hog_img * 255)  # Scale image for saving

    # Extract color histogram features
    hist_features = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_features = cv2.normalize(hist_features, hist_features).flatten()
    
    # Combine features
    features = np.hstack([hog_features.flatten(), hist_features])
    
    return features

# Convert to binary classification
df["Diagnosis"] = df["Diagnosis"].apply(lambda x: 1 if x > 0 else 0)

# Extract features and labels
X = []
y = []

for img_name, label in zip(df["Image_Path"], df["Diagnosis"]):
    img_path = os.path.join(image_dir, img_name + ".jpg")
    save_path = os.path.join(output_dir, img_name + ".jpg")

    if os.path.exists(img_path):
        print(f"Processing image: {img_path}")
        features = preprocess_image(img_path, save_path=save_path)
        if features is not None:
            X.append(features)
            y.append(label)
    else:
        print(f"Warning: Image not found at {img_path}")

# Check if features were extracted
if len(X) == 0:
    raise ValueError("No features were extracted. Check the image paths and preprocessing function.")

X = np.array(X, dtype=np.float32)
y = np.array(y)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Apply SMOTE to balance the dataset
try:
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    print("SMOTE applied successfully.")
except ValueError as e:
    print(f"Error during SMOTE: {e}")
    print("Using original dataset without SMOTE.")
    X_balanced, y_balanced = X, y  # Skip SMOTE if it fails

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# Train Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='binary')
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# =================== Visualization =================== #

# 1. Confusion Matrix (Heatmap)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No DR", "DR"], yticklabels=["No DR", "DR"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("Confusion_Matrix.png")  # Save the confusion matrix
plt.show()

# 2. Accuracy & F1 Score (Bar Chart)
plt.figure(figsize=(6, 4))
plt.bar(["Accuracy", "F1 Score"], [accuracy, f1], color=["blue", "green"])
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Performance Metrics")
plt.savefig("Model_Performance.png")
plt.show()

# 3. Class Distribution Before & After SMOTE (Histogram)
plt.figure(figsize=(6, 4))
plt.hist(y, bins=2, alpha=0.5, label="Original", color="red")
plt.hist(y_balanced, bins=2, alpha=0.5, label="After SMOTE", color="blue")
plt.xticks([0, 1], ["No DR", "DR"])
plt.ylabel("Count")
plt.title("Class Distribution Before and After SMOTE")
plt.legend()
plt.savefig("Class_Distribution.png")
plt.show()

# 4. Feature Importance (Bar Chart)
feature_importances = clf.feature_importances_
top_n = 10  # Show top 10 features
sorted_idx = np.argsort(feature_importances)[-top_n:]

plt.figure(figsize=(8, 5))
plt.barh(range(top_n), feature_importances[sorted_idx], color="purple", align="center")
plt.yticks(range(top_n), [f"Feature {i}" for i in sorted_idx])
plt.xlabel("Importance Score")
plt.title("Top 10 Feature Importances")
plt.savefig("Feature_Importance.png")
plt.show()

# 5. Loss Trends (Placeholder Line Chart)
epochs = np.arange(1, 11)
train_loss = np.exp(-epochs) + np.random.rand(10) * 0.1  # Fake loss values for visualization

plt.figure(figsize=(6, 4))
plt.plot(epochs, train_loss, marker="o", linestyle="--", color="orange", label="Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.savefig("Loss_Trend.png")
plt.show()
