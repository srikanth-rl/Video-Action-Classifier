from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
import cv2
desired_height = 224
desired_width = 224
num_frames = 30
num_channels = 3

class_names = ['Boxing', 'Swimming', 'Jogging']
video_paths = [
    "/content/VID-20240108-WA0004.mp4",
    "/content/VID-20240108-WA0002.mp4",
    "/content/VID-20240108-WA0003.mp4"
]


def preprocess_video(video_path, desired_height, desired_width, num_frames):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (desired_width, desired_height))
            frames.append(frame)

    cap.release()
    return frames


base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(desired_height, desired_width, 3))

for layer in base_model.layers:
    layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(class_names), activation='softmax'))

model.compile(optimizer=Adam(), loss='categorical_crossentropy',
              metrics=['accuracy'])

X = []
y = []

for i, video_path in enumerate(video_paths):
    frames = preprocess_video(
        video_path, desired_height, desired_width, num_frames)
    label = to_categorical(i, len(class_names))

    X.extend(frames)
    y.extend([label] * len(frames))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model.fit(preprocess_input(X_train), y_train, epochs=5, batch_size=32)

y_pred = model.predict(preprocess_input(X_test))
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"Test accuracy: {accuracy * 100:.2f}%")
model.save("/content/video_action_model.h5")
