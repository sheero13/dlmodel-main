import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Step i: Load the dataset
# Note: Replace `YOUR_DATASET_PATH` with the actual path to the horse or human dataset.

data_gen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)  # Normalizing images
train_data = data_gen.flow_from_directory(
    'YOUR_DATASET_PATH', 
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
val_data = data_gen.flow_from_directory(
    'YOUR_DATASET_PATH', 
    target_size=(224, 224),
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

# Step iv: Using 2 Pre-trained CNN models (VGG16 and ResNet50)

# Function to build a model based on a pre-trained network with transfer learning
def build_transfer_model(base_model, learning_rate=0.001):
    base_model.trainable = False  # Freeze base model layers
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),  # Fully connected layer with 128 neurons
        Dropout(0.5),  # Adding dropout for regularization
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

# Load pre-trained VGG16 model
vgg16_base = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
vgg16_model = build_transfer_model(vgg16_base)

# Load pre-trained ResNet50 model
resnet50_base = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
resnet50_model = build_transfer_model(resnet50_base)

# Step iv continued: Experimenting with different parameters
# Experiment 1: Train with VGG16, learning rate = 0.001, batch size = 32
print("Training VGG16 model...")
vgg16_history = vgg16_model.fit(
    train_data,
    validation_data=val_data,
    epochs=5  # Adjust epochs based on model performance
)

# Experiment 2: Train with ResNet50, learning rate = 0.001, batch size = 32
print("Training ResNet50 model...")
resnet50_history = resnet50_model.fit(
    train_data,
    validation_data=val_data,
    epochs=5  # Adjust epochs based on model performance
)

# Step vii: Fine-tuning the models
def fine_tune_model(base_model, model, learning_rate=0.0001):
    base_model.trainable = True  # Unfreeze base model for fine-tuning
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

# Fine-tune VGG16
vgg16_model = fine_tune_model(vgg16_base, vgg16_model)
vgg16_fine_tune_history = vgg16_model.fit(
    train_data,
    validation_data=val_data,
    epochs=3  # Fewer epochs for fine-tuning
)

# Fine-tune ResNet50
resnet50_model = fine_tune_model(resnet50_base, resnet50_model)
resnet50_fine_tune_history = resnet50_model.fit(
    train_data,
    validation_data=val_data,
    epochs=3
)

# Step x: Evaluate models on the test data and display confusion matrix

# Load test data
test_data = data_gen.flow_from_directory(
    'YOUR_DATASET_PATH', 
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Function to evaluate the model and display results
def evaluate_and_report(model, data, title):
    # Evaluate accuracy and loss
    loss, accuracy = model.evaluate(data)
    print(f'{title} - Test Loss: {loss}, Test Accuracy: {accuracy}')
    
    # Generate predictions and confusion matrix
    y_pred = model.predict(data)
    y_pred_classes = (y_pred > 0.5).astype("int32")
    y_true = data.classes

    cm = confusion_matrix(y_true, y_pred_classes)
    print(f"{title} - Confusion Matrix:\n", cm)
    
    # Classification report
    print(f"{title} - Classification Report:\n", classification_report(y_true, y_pred_classes))

# Evaluate and report for VGG16
evaluate_and_report(vgg16_model, test_data, "VGG16")

# Evaluate and report for ResNet50
evaluate_and_report(resnet50_model, test_data, "ResNet50")

# Plot training and fine-tuning history for VGG16 and ResNet50
def plot_history(history, title):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title(f'{title} - Training and validation accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(f'{title} - Training and validation loss')
    plt.legend()
    
    plt.show()

# Plot history for VGG16
plot_history(vgg16_history, "VGG16 Initial Training")
plot_history(vgg16_fine_tune_history, "VGG16 Fine-Tuning")

# Plot history for ResNet50
plot_history(resnet50_history, "ResNet50 Initial Training")
plot_history(resnet50_fine_tune_history, "ResNet50 Fine-Tuning")