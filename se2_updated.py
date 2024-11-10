from google.colab import drive
drive.mount('/content/drive')

dataset_path = '/content/drive/My Drive/horse-or-human/horse-or-human'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

data_gen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2) 
train_data = data_gen.flow_from_directory(
    dataset_path, 
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
val_data = data_gen.flow_from_directory(
    dataset_path, 
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

def plot_sample_images(data, labels):
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        img, label = next(data)
        plt.imshow(img[0])
        plt.title(f'Label: {label[0]}')
        plt.axis('off')
    plt.show()

plot_sample_images(train_data, train_data.classes)

def build_transfer_model(base_model, learning_rate=0.001):
    base_model.trainable = False
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

vgg16_base = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
vgg16_model = build_transfer_model(vgg16_base)

resnet50_base = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
resnet50_model = build_transfer_model(resnet50_base)

print("Training VGG16 model...")
vgg16_history = vgg16_model.fit(
    train_data,
    validation_data=val_data,
    epochs=5  
)

print("Training ResNet50 model...")
resnet50_history = resnet50_model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

def fine_tune_model(base_model, model, learning_rate=0.0001):
    base_model.trainable = True
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

vgg16_model = fine_tune_model(vgg16_base, vgg16_model)
vgg16_fine_tune_history = vgg16_model.fit(
    train_data,
    validation_data=val_data,
    epochs=3 
)

resnet50_model = fine_tune_model(resnet50_base, resnet50_model)
resnet50_fine_tune_history = resnet50_model.fit(
    train_data,
    validation_data=val_data,
    epochs=3
)

test_data = data_gen.flow_from_directory(
    dataset_path, 
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

def evaluate_and_report(model, data, title):
    loss, accuracy = model.evaluate(data)
    print(f'{title} - Test Loss: {loss}, Test Accuracy: {accuracy}')
    
    y_pred = model.predict(data)
    y_pred_classes = (y_pred > 0.5).astype("int32")
    y_true = data.classes

    cm = confusion_matrix(y_true, y_pred_classes)
    print(f"{title} - Confusion Matrix:\n", cm)
    
    print(f"{title} - Classification Report:\n", classification_report(y_true, y_pred_classes))

evaluate_and_report(vgg16_model, test_data, "VGG16")

evaluate_and_report(resnet50_model, test_data, "ResNet50")

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

plot_history(vgg16_history, "VGG16 Initial Training")
plot_history(vgg16_fine_tune_history, "VGG16 Fine-Tuning")

plot_history(resnet50_history, "ResNet50 Initial Training")
plot_history(resnet50_fine_tune_history, "ResNet50 Fine-Tuning")