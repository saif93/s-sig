import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf
from skimage.transform import resize
from tqdm import tqdm_notebook
from keras.preprocessing.image import load_img

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Model Components (High Fidelity Config) ---

def downsample(filters, size, apply_norm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    if apply_norm:
        # Using GroupNormalization as InstanceNorm
        result.add(layers.GroupNormalization(groups=-1)) 
    result.add(layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    result.add(layers.GroupNormalization(groups=-1))
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result

def Generator():
    # Input shape (128, 128, 1)
    inputs = Input(shape=[128, 128, 1])

    # Encoder: 4 layers (128 -> 64 -> 32 -> 16 -> 8)
    # Stopping at 8x8 preserves the spatial structure of the mask better than 2x2
    down_stack = [
        downsample(64, 4, apply_norm=False), # (bs, 64, 64, 64)
        downsample(128, 4),                 # (bs, 32, 32, 128)
        downsample(256, 4),                 # (bs, 16, 16, 256)
        downsample(512, 4),                 # (bs, 8, 8, 512)
    ]

    # Decoder
    up_stack = [
        upsample(256, 4, apply_dropout=True), # (bs, 16, 16, 512)
        upsample(128, 4),                    # (bs, 32, 32, 256)
        upsample(64, 4),                     # (bs, 64, 64, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(1, 4, strides=2, padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh') # (bs, 128, 128, 1)

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)
    return Model(inputs=inputs, outputs=x)

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = Input(shape=[128, 128, 1], name='input_mask')
    tar = Input(shape=[128, 128, 1], name='target_image')

    x = layers.concatenate([inp, tar]) 

    # Multi-scale patch analysis
    d1 = downsample(64, 4, False)(x) 
    d2 = downsample(128, 4)(d1)   
    d3 = downsample(256, 4)(d2)   

    last = layers.Conv2D(1, 4, strides=1, padding='same',
                         kernel_initializer=initializer)(d3) 

    return Model(inputs=[inp, tar], outputs=last)

# --- 2. Loss Functions ---

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
LAMBDA = 150 # Increased structural weight
TV_WEIGHT = 1e-6 # Total Variation weight to reduce noise

def generator_loss(disc_generated_output, gen_output, target):
    # GAN Loss (Adversarial)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    
    # L1 Loss (Structural)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    
    # TV Loss (Smoothing)
    tv_loss = tf.reduce_mean(tf.image.total_variation(gen_output))
    
    total_gen_loss = gan_loss + (LAMBDA * l1_loss) + (TV_WEIGHT * tv_loss)
    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output) * 0.9, disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    return (real_loss + generated_loss) * 0.5

# --- 3. Trainer & Monitor ---

class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, sample_masks, sample_images):
        self.sample_masks = sample_masks
        self.sample_images = sample_images

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 300 == 0:
            display_samples(self.model.generator, self.sample_masks, self.sample_images, epoch+1)

def display_samples(model, test_masks, test_images=None, epoch=0):
    prediction = model(test_masks, training=False)
    n = min(5, test_masks.shape[0])
    plt.figure(figsize=(15, 3 * n))
    for i in range(n):
        imgs = [test_masks[i], test_images[i], prediction[i]]
        titles = ["Input Mask", "Ground Truth", f"Generated (Ep {epoch})"]
        for j in range(3):
            plt.subplot(n, 3, i * 3 + j + 1)
            if i == 0: plt.title(titles[j])
            plt.imshow(imgs[j] * 0.5 + 0.5, cmap='gray')
            plt.axis('off')
    plt.tight_layout()
    plt.show()

class Pix2PixTrainer(Model):
    def __init__(self, generator, discriminator):
        super(Pix2PixTrainer, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, gen_optimizer, disc_optimizer, gen_loss_fn, disc_loss_fn):
        super(Pix2PixTrainer, self).compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn

    @tf.function
    def train_step(self, data):
        input_mask, target_image = data
        
        # Random Augmentation for Diversity
        if tf.random.uniform(()) > 0.5:
            input_mask = tf.image.flip_left_right(input_mask)
            target_image = tf.image.flip_left_right(target_image)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_mask, training=True)
            disc_real_output = self.discriminator([input_mask, target_image], training=True)
            disc_generated_output = self.discriminator([input_mask, gen_output], training=True)

            gen_total_loss, _, _ = self.gen_loss_fn(disc_generated_output, gen_output, target_image)
            disc_loss = self.disc_loss_fn(disc_real_output, disc_generated_output)

        self.gen_optimizer.apply_gradients(zip(gen_tape.gradient(gen_total_loss, self.generator.trainable_variables), self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_tape.gradient(disc_loss, self.discriminator.trainable_variables), self.discriminator.trainable_variables))
        return {"gen_loss": gen_total_loss, "disc_loss": disc_loss}

# --- 4. Execution ---

def run_training(x_train, y_train, epochs=200, batch_size=16):
    x_train_proc = (x_train.astype('float32') / 127.5) - 1.0
    y_train_proc = (y_train.astype('float32') / 127.5) - 1.0

    generator = Generator()
    discriminator = Discriminator()

    # Lower Learning Rate with Adam optimizer to avoid overshooting
    gen_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.Adam(5e-5, beta_1=0.5)

    pix2pix = Pix2PixTrainer(generator, discriminator)
    pix2pix.compile(gen_optimizer, disc_optimizer, generator_loss, discriminator_loss)

    monitor_cb = GANMonitor(y_train_proc[:5], x_train_proc[:5])
    pix2pix.fit(y_train_proc, x_train_proc, epochs=epochs, batch_size=batch_size, callbacks=[monitor_cb])
    
    return generator, x_train_proc, y_train_proc


if name == 'main':
	trained_gen, x_proc, y_proc = run_training(x_train, y_train, epochs=3000, batch_size=32)

