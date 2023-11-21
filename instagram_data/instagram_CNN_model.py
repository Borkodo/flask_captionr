from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import csv
import numpy as np
from keras.models import Model
from keras.layers import Embedding, LSTM, Add, Dense, Input, Flatten
from keras.applications import VGG16
from keras.utils import Sequence
from keras.layers import TimeDistributed
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf

print("Step 1: Model Compilation...")
# Define 'vocab_size' based on your data
vocab_size = 27742  # Replace with the size of your vocabulary
# Define 'max_len' based on your data
max_len = 402  # Replace with the maximum length of your captions

# Define the CNN model (VGG16)
image_input = Input(shape=(224, 224, 3))
vgg = VGG16(input_tensor=image_input, include_top=False, weights='imagenet')
for layer in vgg.layers: layer.trainable = False
image_features_layer = Dense(256, activation='relu')(Flatten()(vgg.output))

# Define the RNN model (LSTM)
caption_input = Input(shape=(max_len,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(caption_input)
lstm_layer = LSTM(units=256, return_sequences=True)(embedding_layer)

# Merge the output of CNN and RNN models
decoder_input = Add()([image_features_layer, lstm_layer])

# Output layer
output = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder_input)

# Final model
model = Model(inputs=[image_input, caption_input], outputs=output)

# Compile the model
try:
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
except Exception as e:
    print(f"Model failed to compile: {e}")
    exit(1)
print("Model summary:")
model.summary()

print("Model compiled.")
print("Expected model output shape:", model.output_shape)

print("Step 2: DataGenerator Class Definition...")


class UnifiedDataGenerator(Sequence):
    def __init__(self, image_paths, sequences, target_labels, batch_size, is_training):
        self.image_paths = image_paths
        self.sequences = sequences
        self.target_labels = target_labels  # This is the memmapped array
        self.batch_size = batch_size
        self.is_training = is_training
        self.indices = np.arange(len(self.image_paths))
        if self.is_training:
            np.random.shuffle(self.indices)

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        index_range = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_image_paths = [self.image_paths[i] for i in index_range]
        batch_sequences = [self.sequences[i] for i in index_range]
        batch_target_labels = [self.target_labels[i] for i in index_range]

        images = []
        for path in batch_image_paths:
            img_path = os.path.join("archive/instagram_data", path + ".jpg")
            img_path = img_path.replace('img2', 'img')
            img = load_img(img_path, target_size=(224, 224))
            img = img_to_array(img)
            img = preprocess_input(img)
            images.append(img)

        return [np.array(images), np.array(batch_sequences)], np.array(batch_target_labels)


print("DataGenerator class defined.")

print("Step 3: Initialize Variables...")
# Initialize variables
image_paths = []
captions = []
sequences = []

print("Reading CSV file...")
with open('archive/combined_file.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader, None)  # Skip the header
    for row in reader:
        try:
            image_path = row[1]  # 2nd column
            caption = row[2]  # 3rd column
            image_paths.append(image_path)
            captions.append(caption)
        except IndexError as e:
            print(f"Skipping row due to error: {e}")
print(f"Read {len(image_paths)} rows from the CSV file.")

print("Preprocessing text...")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(captions)

# Prepare sequences and target labels
input_sequences = [seq[:-1] for seq in sequences]
target_sequences = [seq[1:] for seq in sequences]

input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_len, padding='post')

shape = target_sequences.shape
dtype = target_sequences.dtype

filename = "target_labels.dat"
fp = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
fp[:] = target_sequences[:]

# Assign the memory-mapped array to target_labels
target_labels = fp

print("Text preprocessing done.")

print("Step 7: Data Splitting...")
# Split data into training and validation sets
all_indices = np.arange(len(image_paths))
train_indices, val_indices = train_test_split(all_indices, test_size=0.2, random_state=42)

train_img_paths = [image_paths[i] for i in train_indices]
val_img_paths = [image_paths[i] for i in val_indices]

train_sequences = [input_sequences[i] for i in train_indices]
val_sequences = [input_sequences[i] for i in val_indices]

if len(image_paths) != len(input_sequences) or len(image_paths) != len(target_labels):
    print("Data length mismatch. Exiting.")
    exit(1)

print("Step 8. Create Data Generators")
# Create Data Generators for training and validation sets
train_gen = UnifiedDataGenerator(train_img_paths, train_sequences, target_labels, batch_size=32, is_training=True)
val_gen = UnifiedDataGenerator(val_img_paths, val_sequences, target_labels, batch_size=32, is_training=False)

# Manually get one batch of data
print("Fetching one batch of data for manual model check...")
one_batch = train_gen.__getitem__(0)  # This fetches the first batch
sample_images, sample_sequences = one_batch[0]
sample_target_labels = one_batch[1]

# Print the shapes to make sure they are correct
print("Shapes for manual model check:")
print("Shape of sample images:", sample_images.shape)
print("Shape of sample sequences:", sample_sequences.shape)
print("Shape of sample target labels:", sample_target_labels.shape)

# Manually make a prediction to check output shape
print("Manually checking model output shape...")
model_output = model.predict(x=[sample_images, sample_sequences])
print("Shape of model output:", model_output.shape)

print("Step 9. Train the model")
# Train the model
input_shape = model.layers[0].input_shape[0][1:]  # Getting the input shape of the image
if input_shape != sample_images.shape[1:]:
    print(
        f"Input shapes do not match. Model input shape is {input_shape}, while sample input shape is {sample_images.shape[1:]}")
    exit(1)

try:
    model.fit(train_gen, validation_data=val_gen, epochs=10)
except tf.errors.InvalidArgumentError as e:
    print("Invalid argument error caught:", e)
    exit(1)
except Exception as e:
    print(f"An exception occurred: {e}")
    exit(1)

print("Model trained successfully.")
try:
    model.save("my_model.h5")
except Exception as e:
    print(f"Failed to save the model: {e}")
    exit(1)
