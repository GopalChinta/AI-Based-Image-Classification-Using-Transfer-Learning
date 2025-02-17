import tensorflow as tf
from tensorflow.keras import layers, models

# Load the pre-trained ResNet50 model without the top classification layer
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of ResNet50 to avoid retraining
base_model.trainable = False

# Add custom classification layers on top of the base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # Adjust the number of classes accordingly
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Summary of the model
model.summary()

# Assuming `train_images`, `train_labels`, `val_images`, `val_labels` are your training and validation datasets
# train_images, train_labels = <your_train_data>
# val_images, val_labels = <your_val_data>

# Train the model
history = model.fit(train_images, train_labels, 
                    epochs=10, 
                    validation_data=(val_images, val_labels), 
                    batch_size=32)
