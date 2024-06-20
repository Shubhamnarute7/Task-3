import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

def extract_features(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = base_model.predict(img_array)
    features = features.flatten()  # Flatten to use as SVM input
    return features
import os

def load_data(directory):
    X = []
    y = []
    for filename in os.listdir(directory):
        if filename.startswith("cat"):
            label = 0
        elif filename.startswith("dog"):
            label = 1
        else:
            continue
        img_path = os.path.join(directory, filename)
        features = extract_features(img_path)
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

# Assuming 'train' and 'test' directories exist with images
X_train, y_train = load_data('train')
X_test, y_test = load_data('test')
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize SVM classifier
svm = SVC(kernel='linear', C=1.0, random_state=42)

# Train SVM on extracted features
svm.fit(X_train, y_train)

# Predict on test data
y_pred = svm.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
