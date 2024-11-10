import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Step i: Load the dataset
# Note: Replace `YOUR_DATASET_PATH` with the path to your horse or human dataset.

data_gen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)  # Normalizing images
train_data = data_gen.flow_from_directory(
    'YOUR_DATASET_PATH', 
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
val_data = data_gen.flow_from_directory(
    'YOUR_DATASET_PATH', 
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Step iii: Visualize some samples from the dataset
def plot_sample_images(data, labels):
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        img, label = data.next()
        plt.imshow(img[0])
        plt.title(f'Label: {label[0]}')
        plt.axis('off')
    plt.show()

plot_sample_images(train_data, train_data.classes)

# Step iv: Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Adding dropout for regularization
    Dense(1, activation='sigmoid')  # Binary classification
])

# Step v: Compile the model with an appropriate loss function and optimizer
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Step vii: Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Step viii: Plot training & validation loss and accuracy to monitor over epochs
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()

plot_history(history)

# Step ix: Evaluate the model on the test set
test_data = data_gen.flow_from_directory(
    'YOUR_DATASET_PATH', 
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)
loss, accuracy = model.evaluate(test_data)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Step x: Generate a confusion matrix for test data
y_pred = model.predict(test_data)
y_pred_classes = (y_pred > 0.5).astype("int32")
y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:\n", cm)

# Optional: Print detailed classification report
print("Classification Report:\n", classification_report(y_true, y_pred_classes))

# Step 2: Experiment with different learning rates, batch sizes, and layer sizes
# Example: Modifying the learning rate and re-training
model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Implement regularization to prevent overfitting/underfitting
# (Dropout was already used above; you could also try early stopping)
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stopping]
)
