import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
import os

# Load the trained model
model = tf.keras.models.load_model('path_to_trained_model.h5')

# Load test data (assuming `test_images` and `test_labels` are pre-loaded)
# test_images, test_labels = <your_test_data>

# Function to load and preprocess images
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust size according to model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array

# Predictions on test images
predictions = model.predict(test_images)  # Assuming `test_images` is the dataset

# Convert predictions to labels
predicted_labels = np.argmax(predictions, axis=1)

# Confusion Matrix
cm = confusion_matrix(test_labels, predicted_labels)
print("Confusion Matrix:")
print(cm)

# Classification Report
report = classification_report(test_labels, predicted_labels)
print("\nClassification Report:")
print(report)

# Visualizing the Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))  # Adjust according to your classes
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
