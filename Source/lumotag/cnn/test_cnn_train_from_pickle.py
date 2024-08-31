import numpy as np
import tensorflow as tf
import os
import pickle
# Disable GPU
tf.config.set_visible_devices([], 'GPU')

def files_in_folder(directory, imgtypes: list[str]):
    allFiles = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name[-3:len(name)] in imgtypes:
                allFiles.append(os.path.join(root, name))
    return allFiles

# Generate some dummy data (replace this with your actual data)
def generate_dummy_data(num_samples):
    X = np.random.randn(num_samples, 50)
    y = np.random.randint(2, size=(num_samples, 1))
    return X, y
def create_noise_proof_model(normalize_input=True):
    inputs = tf.keras.layers.Input(shape=(50, 1))
    
    if normalize_input:
        x = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)
    else:
        x = inputs
    
    x = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
# Create the model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(50,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, X, y, epochs=10, batch_size=32):
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Evaluate the model
def evaluate_model(model, X, y):
    loss, accuracy = model.evaluate(X, y)
    print(f"Test accuracy: {accuracy}")

# Main execution
X_train, y_train = generate_dummy_data(10000)
X_test, y_test = generate_dummy_data(1000)


# load in pickle files and extract data:
folder = r"D:\lumotag_training_data"

pickle_files = files_in_folder(folder, [".pc"])
result_pairs = []
for picklefile in pickle_files:
    with open(picklefile, 'rb') as file:
        result_data= pickle.load(file)
    for pair in result_data:
        assert len(pair) == 2
        result_pairs.append(pair)

training_vectors = []
training_vectors_mirrored = []

for pair in result_pairs:
    training_vectors.append(np.concatenate((pair[0], pair[1])))
    training_vectors_mirrored.append(np.concatenate((pair[1], pair[0])))

model = create_noise_proof_model(normalize_input=True)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


train_model(model, X_train, y_train)
evaluate_model(model, X_test, y_test)

# Save the model
model.save('barcode_model.h5')

# Load the model (this is how you'd load it on the Raspberry Pi)
loaded_model = tf.keras.models.load_model('barcode_model.h5')

# Function to estimate FPS
def estimate_fps(model, num_iterations=1000):
    dummy_input = np.random.randn(1, 50)
    
    start_time = tf.timestamp()
    for _ in range(num_iterations):
        _ = model.predict(dummy_input)
    end_time = tf.timestamp()
    
    total_time = end_time - start_time
    fps = num_iterations / total_time
    
    print(f"Estimated FPS: {fps:.2f}")

# Estimate FPS
estimate_fps(loaded_model)