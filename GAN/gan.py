import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
import random

# Generate a synthetic dataset if the file doesn't exist
def generate_realistic_data(num_records=600):
    activities = ['Homework', 'Video Games', 'Reading', 'TV', 'Exercise', 'Music', 'Social Media', 'Cooking', 'Sports']
    genders = ['M', 'F']
    locations = ['Home', 'Library', 'Community Center', 'Friend\'s House', 'Park', 'Gym']
    interest_levels = ['Low', 'Medium', 'High']
    age_range = range(12, 18)
    performance_scores = range(30, 101)

    data = {
        'student_id': range(1, num_records + 1),
        'activity_type': [random.choice(activities) for _ in range(num_records)],
        'duration': [random.randint(30, 180) for _ in range(num_records)],
        'age': [random.choice(age_range) for _ in range(num_records)],
        'gender': [random.choice(genders) for _ in range(num_records)],
        'location': [random.choice(locations) for _ in range(num_records)],
        'performance_score': [random.choice(performance_scores) for _ in range(num_records)],
        'interest_level': [random.choice(interest_levels) for _ in range(num_records)],
    }

    # Add additional columns
    data['study_hours'] = np.random.randint(0, 5, size=num_records)
    data['parental_support'] = np.random.choice(['Low', 'Medium', 'High'], size=num_records, p=[0.2, 0.5, 0.3])
    data['peer_influence'] = np.random.choice(['Negative', 'Neutral', 'Positive'], size=num_records, p=[0.3, 0.4, 0.3])
    data['stress_level'] = np.random.randint(1, 10, size=num_records)

    return pd.DataFrame(data)

# Save the dataset if it doesn't exist
try:
    real_data = pd.read_csv('realistic_evening_activities.csv')
except FileNotFoundError:
    print("File 'realistic_evening_activities.csv' not found. Generating a new dataset...")
    real_data = generate_realistic_data(600)
    real_data.to_csv('realistic_evening_activities.csv', index=False)
    print("Dataset saved as 'realistic_evening_activities.csv'.")

# Add new columns to the real dataset
real_data['study_hours'] = np.random.randint(0, 5, size=len(real_data))  # Hours spent studying daily
real_data['parental_support'] = np.random.choice(['Low', 'Medium', 'High'], size=len(real_data), p=[0.2, 0.5, 0.3])
real_data['peer_influence'] = np.random.choice(['Negative', 'Neutral', 'Positive'], size=len(real_data), p=[0.3, 0.4, 0.3])
real_data['stress_level'] = np.random.randint(1, 10, size=len(real_data))  # Stress level on a scale of 1-10

# Normalize numerical columns for GAN training
numerical_columns = ['duration', 'age', 'performance_score', 'study_hours', 'stress_level']
real_data_normalized = real_data[numerical_columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Convert categorical columns to one-hot encoding
categorical_columns = ['activity_type', 'gender', 'location', 'interest_level', 'parental_support', 'peer_influence']
real_data_encoded = pd.get_dummies(real_data[categorical_columns])

# Combine normalized numerical and encoded categorical data
gan_input_data = pd.concat([real_data_normalized, real_data_encoded], axis=1).values.astype(np.float32)

# Define GAN parameters
latent_dim = 100  # Size of the random noise vector
data_dim = gan_input_data.shape[1]  # Number of features in the dataset

# Build the Generator
def build_generator():
    model = tf.keras.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(data_dim, activation='sigmoid')  # Output matches the normalized data
    ])
    return model

# Build the Discriminator
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Input(shape=(data_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output is a probability (real or fake)
    ])
    return model

# Compile the GAN
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Combine Generator and Discriminator into a GAN
discriminator.trainable = False
gan_input = layers.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Train the GAN
def train_gan(epochs=10000, batch_size=32):
    for epoch in range(epochs):
        # Train the Discriminator
        real_samples = gan_input_data[np.random.randint(0, gan_input_data.shape[0], batch_size)]
        fake_samples = generator.predict(np.random.normal(0, 1, (batch_size, latent_dim)))
        labels_real = np.ones((batch_size, 1))
        labels_fake = np.zeros((batch_size, 1))
        d_loss_real = discriminator.train_on_batch(real_samples, labels_real)
        d_loss_fake = discriminator.train_on_batch(fake_samples, labels_fake)

        # Train the Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        labels_gan = np.ones((batch_size, 1))  # Generator wants the discriminator to think fake data is real
        g_loss = gan.train_on_batch(noise, labels_gan)

        # Print progress
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, D Loss Real: {d_loss_real[0]}, D Loss Fake: {d_loss_fake[0]}, G Loss: {g_loss}")

# Train the GAN
train_gan(epochs=10000, batch_size=64)

# Generate Synthetic Data
def generate_synthetic_data(num_samples=600):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    synthetic_data = generator.predict(noise)
    synthetic_data = pd.DataFrame(synthetic_data, columns=real_data_normalized.columns.tolist() + real_data_encoded.columns.tolist())
    return synthetic_data

# Generate and save synthetic data
synthetic_data = generate_synthetic_data(600)
synthetic_data.to_csv('gan_synthetic_evening_activities_with_more_columns.csv', index=False)
print("Synthetic dataset created using GAN and saved as 'gan_synthetic_evening_activities_with_more_columns.csv'.")