#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import zeros, ones
from numpy.random import randn, normal
from numpy.random import randint
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization
from matplotlib import pyplot
import os
from PIL import Image

# Define the discriminator model
def define_discriminator(in_shape=(256, 256, 3)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    opt = Adam(learning_rate=0.00005, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# Define the generator model
def define_generator(latent_dim):
    model = Sequential()
    n_nodes = 128 * 8 * 8
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(Reshape((8, 8, 128)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same',activation='relu'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same',activation='relu'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same',activation='relu'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(8, (4, 4), strides=(2, 2), padding='same',activation='relu'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='tanh'))
    return model

# Define the combined generator and discriminator model
def define_gan(g_model, d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

# Load and prepare images from the dataset
def load_real_samples(data_dir="/N/slate/athshah/SNOW_Breast_cancer/train"):
    trainX = []
    for i in os.listdir(data_dir):
        if str(i)[-3:] == "png":
            image = Image.open(os.path.join(data_dir, i))
            image = image.resize((256, 256))
            image_array = np.array(image)
            trainX.append(image_array)

    trainX = np.array(trainX)
    X = trainX.astype('float32')
    X = (X - 127.5) / 127.5
    return X

# Generate real samples with labels
def generate_real_samples(dataset, n_samples):
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = ones((n_samples, 1))
    return X, y

# Generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    x_input = normal(0, 3, (n_samples, latent_dim))
    return x_input

# Generate fake samples with labels using the generator
def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input, verbose=0)
    y = zeros((n_samples, 1))
    return X, y

# Create and save a plot of generated images
def save_plot(examples, epoch, n=2):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        pyplot.subplot(n, n, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(examples[i * 10])
    filename = 'images_new/generated_plot_e%03d.png' % (epoch + 1)
    pyplot.savefig(filename)
    pyplot.show()

# Evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, acc_re, acc_fa, n_samples=150):
    X_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)

    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)

    print(f'>Epoch {epoch + 1}, Accuracy Real: {acc_real * 100:.2f}%, Accuracy Fake: {(1-acc_fake) * 100:.2f}%')

    acc_re.append(acc_real)
    acc_fa.append(1 - acc_fake)

    if epoch > 0 and (epoch + 1) % 1 == 0:
        save_plot(x_fake, epoch)
        filename = f'generator_model_DCGAN_e{epoch + 1}.h5'
        g_model.save(filename)

# Train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=50):
    acc_real_list, acc_fake_list = [], []
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)

    for i in range(n_epochs):
        for j in range(bat_per_epo):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)

            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)

            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)

            if j % 100 == 0:
                print(f'>Epoch {i + 1}, Batch {j + 1}/{bat_per_epo}, D1 Loss: {d_loss1:.3f}, D2 Loss: {d_loss2:.3f}, G Loss: {g_loss:.3f}')

        summarize_performance(i, g_model, d_model, dataset, latent_dim, acc_real_list, acc_fake_list)

# Size of the latent space
latent_dim = 100

# Create the discriminator
d_model = define_discriminator()

# Create the generator
g_model = define_generator(latent_dim)

# Create the GAN
gan_model = define_gan(g_model, d_model)

# Load image data
dataset = load_real_samples()

# Train the model
train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=64)
