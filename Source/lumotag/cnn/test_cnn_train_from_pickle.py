import numpy as np
import tensorflow as tf
import os
import pickle

import os
modelpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "barcode_model.h5")
pickle_pairs_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), "pickledpairs.pc")
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
def train_model(model, X, y, epochs=3, batch_size=32):
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


try:
    pickle_files = files_in_folder(folder, [".pc"])
    result_pairs = []
    for picklefile in pickle_files:
        with open(picklefile, 'rb') as file:
            result_data= pickle.load(file)
        for pair in result_data:
            assert len(pair) == 2
            result_pairs.append(pair)
    if len(result_pairs) < 1:
        raise Exception
    with open(pickle_pairs_path, 'wb') as file:
        pickle.dump(result_pairs, file)
    with open(pickle_pairs_path, 'rb') as file:
        check_data= pickle.load(file)
    plop=1
except Exception as e:
    input("error finding pair files - will load previously saved")

    with open(pickle_pairs_path, 'rb') as file:
        pickle_files= pickle.load(result_pairs)


def split_for_training_and_eval(numpyarray):
    return numpyarray[:int(len(numpyarray)*0.8)], numpyarray[int(len(numpyarray)*0.8):]


player1_class =0
noise_class = 1
training_vectors = []
training_vectors_mirrored = []
total_training_player1 = []
for pair in result_pairs:
    training_vectors.append(np.concatenate((pair[0], pair[1])))
    training_vectors_mirrored.append(np.concatenate((pair[1], pair[0])))
X_training_player1 = np.asarray(training_vectors + training_vectors_mirrored)
y_training_player1 = np.full((len(X_training_player1),1),player1_class)
X_noise_class = np.random.randint(0, 255, size=(len(X_training_player1), 50))
y_noise_class = np.full((len(X_noise_class),1),noise_class)



#splits
X_player_train, X_player_eval = split_for_training_and_eval(X_training_player1)
y_player_train, y_player_eval = split_for_training_and_eval(y_training_player1)

X_noise_train, X_noise_eval = split_for_training_and_eval(X_noise_class)
y_noise_train, y_noise_eval = split_for_training_and_eval(y_noise_class)

X_player_with_noise_train = np.concatenate((X_player_train, X_noise_train))
y_player_with_noise_train = np.concatenate((y_player_train, y_noise_train))

X_player_with_noise_eval = np.concatenate((X_player_eval, X_noise_eval))
y_player_with_noise_eval = np.concatenate((y_player_eval, y_noise_eval))

model = create_noise_proof_model(normalize_input=True)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

x,y = generate_dummy_data(1000)
train_model(model, X_player_with_noise_train, y_player_with_noise_train)
evaluate_model(model, X_player_with_noise_eval, y_player_with_noise_eval)

# Save the model
model.save(modelpath)

# Load the model (this is how you'd load it on the Raspberry Pi)
loaded_model = tf.keras.models.load_model(modelpath)

# # Function to estimate FPS
# def estimate_fps(model, num_iterations=1000):
#     dummy_input = np.random.randn(1, 50)
#     #X_player_with_noise_eval[0].reshape(1,len(X_player_with_noise_eval[0]))
#     start_time = tf.timestamp()
#     for _ in range(num_iterations):
#         _ = model.predict(X_noise_eval[0].reshape(1,len(X_noise_eval[0])))
#     end_time = tf.timestamp()

#     total_time = end_time - start_time
#     fps = num_iterations / total_time
   
#     print(f"Estimated FPS: {fps:.2f}")

# # Estimate FPS
# estimate_fps(model)