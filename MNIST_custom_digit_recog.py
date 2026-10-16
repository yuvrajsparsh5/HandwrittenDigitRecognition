import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import os

print("Step 1: Loading MNIST Data...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("Step 2: Building and Training the AI Model (This will take a few seconds)...")
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

def predict_custom_image(image_path, trained_model):
    try:
        if not os.path.exists(image_path):
            print(f"Error: Could not find the image at '{image_path}'. Please check the file path.")
            return

        img = Image.open(image_path).convert('L')
        
        if np.mean(img) > 127:
            img = ImageOps.invert(img)
            
        img = img.resize((28, 28))
        
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28)
        
        predictions = trained_model.predict(img_array)
        predicted_digit = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        
        print(f"\n>>> AI Predicts: {predicted_digit} (Confidence: {confidence:.2f}%)")
        
        plt.imshow(img_array[0], cmap='gray')
        plt.title(f"AI Predicts: {predicted_digit} (Confidence: {confidence:.2f}%)")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

print("\n" + "="*50)
print("--- AI Ready for Testing ---")
print("Tip: In Windows, you can Right-Click a file and select 'Copy as path', then paste it here.")
print("="*50)



while True:
    user_input = input("\nEnter image path (or type 'exit' to quit): ")
    
    if user_input.lower() == 'exit':
        print("Closing the program. Goodbye!")
        break
        
    clean_path = user_input.strip('"').strip("'")
    
    if clean_path:
        predict_custom_image(clean_path, model)
