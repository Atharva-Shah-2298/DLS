
import pickle
import tensorflow as tf
import numpy as np
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, MaxPool2D, Reshape, Conv2DTranspose
from keras.models import Sequential
from keras import Model
from keras import backend as K
from keras.models import load_model
from tensorflow import keras
from keras import layers
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from keras.optimizers import Adam
from keras.metrics import mean_squared_error
from matplotlib import pyplot as plt
from tqdm import tqdm
from keras import metrics

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
# Define hyperparameters

epochs = 100

from PIL import Image
import numpy as np

print("Imported libraries")
def load_image(img_path):
    try:
        # Open image using PIL
        img = Image.open(img_path)
        
        # Resize the image to the target size
        img = img.resize((512, 512))
        
        # Convert the image to a NumPy array
        img_array = np.array(img) / 255.0  # Normalize pixel values to be between 0 and 1
        
        return img_array
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None
    
latent_dim = 500  # Set to the desired number of latent dimensions
encoder_inputs = layers.Input(shape=(512, 512, 3))
x = encoder_inputs
x = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', name='layer1')(x)
x = tf.keras.layers.MaxPooling2D((2, 2), name='maxpool1')(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='layer2')(x)
x = tf.keras.layers.MaxPooling2D((2, 2), name='maxpool2')(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='layer3')(x)
x = tf.keras.layers.MaxPooling2D((2, 2), name='maxpool3')(x)
x = tf.keras.layers.Flatten(name='layer4')(x)
z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)

# Sampling logic integrated into the model
batch = tf.shape(z_mean)[0]
dim = tf.shape(z_mean)[1]
epsilon = K.random_normal(shape=(batch, dim))
z = z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

# Build the encoder model
encoder_model = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

# Print the summary of the modified encoder model
encoder_model.summary()

latent_dim = 512  # Update this based on your desired latent dimension

decoder_model = tf.keras.Sequential([
    layers.Dense(latent_dim, activation='relu', name='layer1'),
    layers.Dense(128*128*3, activation='relu', name='layer2'),
    layers.Reshape(target_shape=(128, 128, 3)),
    layers.Conv2DTranspose(128, (3, 3), activation='relu', name='layer3', padding='same'),
    layers.UpSampling2D((2, 2), name='upsample1'),
    layers.Conv2DTranspose(64, (3, 3), activation='relu', name='layer4', padding='same'),
    layers.UpSampling2D((2, 2), name='upsample2'),
    layers.Conv2DTranspose(3, (5, 5), activation='sigmoid', name='layer5', padding='same'),
])

decoder_output = decoder_model(z)

decoder_model.summary()

vae = Model(encoder_inputs, decoder_output)

print("Defined the model")
original_dim=512*512

xent_loss = original_dim*metrics.mean_squared_error(K.flatten(encoder_inputs), K.flatten(decoder_output))
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
elbo_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(elbo_loss)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001))

batch_size = 32
epochs = 2
folder_path = '/N/slate/athshah/Breast_cancer/Image'

# Get the list of image files in the directory
file_names = [file_name for file_name in os.listdir(folder_path) if ".ipynb_checkpoints" not in file_name]

print(len(file_names))
num_samples = 10

# Training loop
loss_curve = []
for epoch in range(epochs):
    for batch_files in np.array_split(file_names, len(file_names) // batch_size):
        batch_images = [load_image(os.path.join(folder_path, file_name)) for file_name in batch_files]
        batch_images = np.array(batch_images)

        # Train your VAE model with the current batch
        history  = vae.fit(batch_images, epochs=1, batch_size=batch_size, verbose=1)
        loss_curve.append(history.history["loss"])
    print(f'Epoch {epoch + 1} complete')

    # Choose a random batch for visualization
    random_batch_files = np.random.choice(file_names, size=num_samples, replace=False)
    test_original = [load_image(os.path.join(folder_path, file_name)) for file_name in random_batch_files]
    test_original = np.array(test_original)

    test_rec = vae.predict(test_original)

 # Visualization code...
for i in range(num_samples):
    plt.figure(figsize=(10, 5))  # Adjust the figure size as needed

    # Subplot 1: Original Image
    ax = plt.subplot(1, 2, 1)
    plt.imshow(test_original[i])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    # Subplot 2: Reconstructed Image
    ax = plt.subplot(1, 2, 2)
    plt.imshow(test_rec[i])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the subplot figure
    plt.savefig(f'/N/slate/athshah/VAE_results/epoch_{epoch + 1}_sample_{i}_reconstruction.png')
    plt.close()  # Close the current figure before moving to the next iteration

# Create a new figure for the loss plot
plt.figure(figsize=(8, 5))  # Adjust the figure size as needed

# Plot the loss curve
plt.plot(np.array(loss_curve))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')

# Save the figure for the loss plot
plt.savefig('/N/slate/athshah/VAE_results/loss_plot.png')
    
    #if epoch % 5 == 0:
		
       # # Save encoder and decoder models
       # encoder_model.save(f'/N/slate/athshah/VAE_results/encoder_model_epoch_{epoch + 1}.h5')
       # decoder_model.save(f'/N/slate/athshah/VAE_results/decoder_model_epoch_{epoch + 1}.h5')
		
