import numpy as np
import tensorflow as tf
import os
import pickle
import random
import os
from tensorflow.keras import regularizers
modelpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "barcode_model.h5")
pickle_pairs_path_player1 =  os.path.join(os.path.dirname(os.path.abspath(__file__)), "player1_pickles")
pickle_pairs_path_false_positives =  os.path.join(os.path.dirname(os.path.abspath(__file__)), "false_positive_pickles")
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


def create_noise_proof_non_overfit_model(normalise_input=True):
    inputs = tf.keras.layers.Input(shape=(50, 1))
    
    if normalize_input:
        x = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)
    else:
        x = inputs
    
    x = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    
    x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(32, activation='relu',
                              kernel_regularizer=regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

import tensorflow as tf

def create_ultra_fast_barcode_cnn_model(normalise_input=True):
    model = tf.keras.Sequential([
        # Normalization layer
        tf.keras.layers.Input(shape=(50, 1)),
        tf.keras.layers.Rescaling(1./255),

        # Use a single Separable Conv layer with fewer filters
        tf.keras.layers.SeparableConv1D(16, kernel_size=3, activation='relu', padding='same'),

        # Increase the pooling size to aggressively reduce the feature map size
        tf.keras.layers.MaxPooling1D(pool_size=4),

        # Global pooling to flatten the output
        tf.keras.layers.GlobalAveragePooling1D(),

        # Further reduce the size of the dense layer
        tf.keras.layers.Dense(8, activation='relu'),

        # Output layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_optimized_barcode_cnn_model(normalise_input=True):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(50, 1)),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.SeparableConv1D(32, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),  # Batch Normalization for faster convergence
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        tf.keras.layers.SeparableConv1D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
def create_barcode_cnn_model(normalise_input=True):
    model = tf.keras.Sequential([
        # Normalization layer to scale input values to the [0, 1] range
        tf.keras.layers.Input(shape=(50, 1)),  # Input shape matches the length of the barcode
        tf.keras.layers.Rescaling(1./255),  # Rescale the pixel values from [0, 255] to [0, 1]
        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.GlobalAveragePooling1D(),  # Summarize features from convolutional layers
        tf.keras.layers.Dense(32, activation='relu'),  # Dense layer to learn more complex features
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer (binary classification or regression)
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Assuming classification
    return model
def create_noise_proof_model(normalise_input=True):
    inputs = tf.keras.layers.Input(shape=(50, 1))
    
    if normalise_input:
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
def train_model(model, X, y, epochs=1000, batch_size=100):
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Evaluate the model
def evaluate_model(model, X, y):
    loss, accuracy = model.evaluate(X, y)
    print(f"Test accuracy: {accuracy}")

# # Main execution
# X_train, y_train = generate_dummy_data(10000)
# X_test, y_test = generate_dummy_data(1000)


# load in pickle files and extract data:
folder = r"D:\lumotag_training_data"



def get_tagged_aggregate_pickle_files(tag: str) -> list:
    folder = os.path.dirname(os.path.abspath(__file__))
    pickle_files = files_in_folder(folder, [".pc"])
    result_pairs = []
    for picklefile in pickle_files:
        if tag not in picklefile:
            continue
        with open(picklefile, 'rb') as file:
            result_data= pickle.load(file)
        for pair in result_data:
            assert len(pair) == 2
            result_pairs.append(pair)
    return result_pairs


def split_for_training_and_eval(numpyarray):
    return numpyarray[:int(len(numpyarray)*0.5)], numpyarray[int(len(numpyarray)*0.5):]



result_pairs = get_tagged_aggregate_pickle_files(tag="player1")
false_positives = get_tagged_aggregate_pickle_files(tag="false")

random.shuffle(result_pairs)
random.shuffle(false_positives)


player1_class = 1
noise_class = 0
training_vectors = []
training_vectors_mirrored = []

training_false_positive_augmented = []

total_training_player1 = []
for pair in result_pairs:
    training_vectors.append(np.vstack((pair[0], pair[1])))
    #training_vectors.append(np.concatenate((pair[0], pair[1])))
    training_vectors_mirrored.append(np.vstack((pair[1], pair[0])))
X_training_player1 = np.asarray(training_vectors + training_vectors_mirrored)
X_training_player1 = X_training_player1
y_training_player1 = np.full((len(X_training_player1),1),player1_class)
X_noise_class = []
for i in range(0,len(y_training_player1)):
    X_noise_class.append(np.vstack((np.random.randint(0, 255, 25), np.random.randint(0, 255, 25))))

#X_noise_class = X_noise_class[0:10] # temp - try removing all noise
# add false positives
for pair in false_positives:
    training_false_positive_augmented.append(np.vstack((pair[0], pair[1])))
    training_false_positive_augmented.append(np.roll(training_false_positive_augmented[-1], shift=len(pair)//2))
    training_false_positive_augmented.append(np.vstack((pair[1], pair[0])))
    training_false_positive_augmented.append(np.roll(training_false_positive_augmented[-1], shift=len(pair)//2))
training_false_positive_augmented = np.asarray(training_false_positive_augmented)
X_noise_class = np.concatenate((X_noise_class, training_false_positive_augmented))
y_noise_class = np.full((len(X_noise_class),1),noise_class)

# convert to uint8 maybs
X_training_player1 = X_training_player1.astype("uint8")
y_training_player1 = y_training_player1.astype("uint8")
assert len(X_training_player1) ==  len(y_training_player1)
X_noise_class = X_noise_class.astype("uint8")
y_noise_class = y_noise_class.astype("uint8")
assert len(X_noise_class) ==  len(y_noise_class)

testa, testb = split_for_training_and_eval(np.asarray([i for i in range(0,10)]))
assert set(testa.tolist()).intersection(set(testb.tolist())) == set()

#splits
X_player_train, X_player_eval = split_for_training_and_eval(X_training_player1)
y_player_train, y_player_eval = split_for_training_and_eval(y_training_player1)

assert len(X_player_train) ==  len(y_player_train)
assert len(X_player_eval) ==  len(y_player_eval)

X_noise_train, X_noise_eval = split_for_training_and_eval(X_noise_class)
y_noise_train, y_noise_eval = split_for_training_and_eval(y_noise_class)

assert len(X_noise_train) ==  len(y_noise_train)
assert len(X_noise_eval) ==  len(y_noise_eval)

X_player_with_noise_train = np.concatenate((X_player_train, X_noise_train))
y_player_with_noise_train = np.concatenate((y_player_train, y_noise_train))

assert len(X_player_with_noise_train) ==  len(y_player_with_noise_train)

X_player_with_noise_eval = np.concatenate((X_player_eval, X_noise_eval))
y_player_with_noise_eval = np.concatenate((y_player_eval, y_noise_eval))

assert len(X_player_with_noise_eval) ==  len(y_player_with_noise_eval)


model = create_noise_proof_model(normalise_input=True)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

if not os.path.exists(modelpath):
    train_model(model, X_player_with_noise_train, y_player_with_noise_train)
    model.save(modelpath)

model = tf.keras.models.load_model(modelpath)


print("evaluate mix")
evaluate_model(model, X_player_with_noise_eval, y_player_with_noise_eval)
print("Evaluate player codes")
evaluate_model(model, X_player_eval, y_player_eval)
print("evaluate noise and false positives")
evaluate_model(model, X_noise_eval, y_noise_eval)
#print("evaluate model with bad classification (should be accuracy of 0)")
#evaluate_model(model, X_player_eval, y_noise_eval[:len(X_player_eval)])

print("noise", model.predict(X_noise_eval[0].reshape(1,len(X_noise_eval[0]))))

print("more noise", model.predict(X_noise_eval[0].reshape(1,len(X_noise_eval[0]))))
print("valid P1", model.predict(X_player_eval[0].reshape(1,len(X_player_eval[0]))))









import cv2

while True:
    test_pair = random.choice([
        [random.choice(training_false_positive_augmented), "false positives"],
        [random.choice(X_player_eval), "player_id"],
        [random.choice(X_noise_class), "noise"]
    ])
    score = model.predict(test_pair[0].reshape(1,len(test_pair[0])))
    print(f"{test_pair[1]} : {score}")
    out_img1 = cv2.resize(np.asarray(test_pair[0][:len(test_pair[0])//2].astype("uint8")), (200, 500), interpolation=cv2.INTER_NEAREST)
    out_img1 = cv2.cvtColor(out_img1, cv2.COLOR_GRAY2BGR)
    out_img2 = cv2.resize(np.asarray(test_pair[0][len(test_pair[0])//2:].astype("uint8")), (200, 500), interpolation=cv2.INTER_NEAREST)
    out_img2 = cv2.cvtColor(out_img2, cv2.COLOR_GRAY2BGR)


    midimg = np.zeros(out_img1.shape, np.uint8)
    if score > 0.95:
        midimg[:,:,1] = 255
    else:
        midimg[:,:,2] = 255 

    stacked_img = np.hstack((
        out_img1,
        midimg,
        out_img2))
    cv2.imshow('graycsale image',stacked_img)
    


    key=cv2.waitKey(0)
    if key == 27:#if ESC is pressed, exit loop
        cv2.destroyAllWindows()
        break

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