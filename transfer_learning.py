import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_images(data_dir, target_size=(224, 224)):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(data_dir, target_size=target_size, class_mode="categorical", shuffle=True)
    return generator

def parse_json(data_dir, sub_dir):
    label_dir = os.path.join(data_dir, sub_dir)
    gesture_labels = ['dislike', 'like', 'ok', 'peace', 'stop', 'no_gesture']
    data = {}
    
    for label in gesture_labels:
        if label == 'no_gesture':
            continue
        json_file = os.path.join(label_dir, f"{label}.json")
        with open(json_file, "r") as f:
            json_data = json.load(f)
            for key, value in json_data.items():
                data[key] = {
                    "boxes": value["bboxes"],
                    "labels": gesture_labels.index(value["labels"][0]),
                    "landmarks": value["landmarks"],
                    "leading_hand": 0 if value["leading_hand"] == "left" else 1
                }
    return data

def create_data_generator(images, json_data, batch_size=32):
    while True:
        batch_indices = np.random.choice(len(images), batch_size)
        batch_images = np.array([images[i] for i in batch_indices])
        batch_json_data = [json_data[str(i)] for i in batch_indices]
        
        batch_boxes = np.array([data["boxes"][0] for data in batch_json_data])
        batch_labels = to_categorical([data["labels"] for data in batch_json_data], num_classes=6)
        batch_landmarks = np.array([np.array(data["landmarks"][0]).flatten() for data in batch_json_data])
        batch_leading_hand = to_categorical([data["leading_hand"] for data in batch_json_data], num_classes=2)
        
        yield batch_images, [batch_boxes, batch_labels, batch_landmarks, batch_leading_hand]

def create_model():
    base_model = MobileNetV2(weights="imagenet", include_top=False)

    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)

    bbox_output = Dense(4, activation="linear", name="bboxes")(x)
    label_output = Dense(6, activation="softmax", name="labels")(x)
    landmarks_output = Dense(42, activation="linear", name="landmarks")(x)
    leading_hand_output = Dense(2, activation="softmax", name="leading_hand")(x)

    outputs = Concatenate()([bbox_output, label_output, landmarks_output, leading_hand_output])

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    return model

train_images = load_and_preprocess_images("hand_gestures/train")
val_images = load_and_preprocess_images("hand_gestures/validation")
train_json_data = parse_json("labels", "train")
val_json_data = parse_json("labels", "validation")

train_generator = create_data_generator(train_images, train_json_data)
val_generator = create_data_generator(val_images, val_json_data)

model = create_model()

epochs = 10
steps_per_epoch = len(train_images) // 32
validation_steps = len(val_images) // 32

history = model.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps
)

model.save("gesture_recognition_model.h5")