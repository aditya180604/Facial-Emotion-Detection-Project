import numpy as np
import os
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import cv2

# Define the emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the dataset
train_dir = 'C:\\Users\\msadi\\Desktop\\AIPROJECT\\dataset\\train'
test_dir = 'C:\\Users\\msadi\\Desktop\\AIPROJECT\\dataset\\test'

# Load the images and labels
train_images = []
train_labels = []
test_images = []
test_labels = []

# Function to match the correct capitalization of labels
def get_emotion_label(label):
    for emotion in emotions:
        if label.lower() == emotion.lower():
            return emotion
    raise ValueError(f"Label {label} not found in emotions list")

# Collect and print unique labels for debugging
train_labels_found = set()
test_labels_found = set()

# Debug statement to check the train directory
print(f"Train directory: {train_dir}")
print(f"Test directory: {test_dir}")

for dir_path, dir_names, file_names in os.walk(train_dir):
    for file_name in file_names:
        img_path = os.path.join(dir_path, file_name)
        try:
            img = load_img(img_path, target_size=(48, 48))
            img_array = img_to_array(img)
            train_images.append(img_array)
            label = dir_path.split(os.sep)[-1]
            train_labels_found.add(label)
            train_labels.append(emotions.index(get_emotion_label(label)))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

for dir_path, dir_names, file_names in os.walk(test_dir):
    for file_name in file_names:
        img_path = os.path.join(dir_path, file_name)
        try:
            img = load_img(img_path, target_size=(48, 48))
            img_array = img_to_array(img)
            test_images.append(img_array)
            label = dir_path.split(os.sep)[-1]
            test_labels_found.add(label)
            test_labels.append(emotions.index(get_emotion_label(label)))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

# Print found labels to debug
print(f"Train labels found: {train_labels_found}")
print(f"Test labels found: {test_labels_found}")

# Check if any images were loaded
if not train_images or not train_labels:
    raise ValueError("No training data loaded. Please check the dataset paths and contents.")
if not test_images or not test_labels:
    raise ValueError("No test data loaded. Please check the dataset paths and contents.")

# Convert the lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Normalize the pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# One-hot encode the labels
train_labels = to_categorical(train_labels, num_classes=len(emotions))
test_labels = to_categorical(test_labels, num_classes=len(emotions))

# Print shapes to debug
print(f'Train images shape: {train_images.shape}')
print(f'Train labels shape: {train_labels.shape}')
print(f'Test images shape: {test_images.shape}')
print(f'Test labels shape: {test_labels.shape}')

# Split the data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(emotions), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc:.2f}')

# Load the Haar cascade for face detection

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Loop through each face detected
        for (x, y, w, h) in faces:
            # Extract the face region from the original frame
            face = frame[y:y+h, x:x+w]

            # Resize the face region to 48x48
            face_resized = cv2.resize(face, (48, 48))

            # Normalize the pixel values to be between 0 and 1
            face_resized = face_resized / 255.0

            # Reshape the face region to (1, 48, 48, 3) for the model
            face_resized = face_resized.reshape((1, 48, 48, 3))

            # Make predictions on the face region
            predictions = model.predict(face_resized)

            # Get the index of the highest probability emotion
            emotion_index = np.argmax(predictions[0])

            # Get the emotion label
            emotion_label = emotions[emotion_index]

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display the emotion label above the face
            cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the output
        cv2.imshow('Facial Emotion Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()