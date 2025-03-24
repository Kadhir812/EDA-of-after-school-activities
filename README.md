# GAN-Based Synthetic Dataset Generator

This project implements a **Generative Adversarial Network (GAN)** to generate synthetic datasets for exploratory data analysis (EDA). The GAN is trained on a realistic dataset and generates new data with similar patterns and features.

## Features

- Generates synthetic datasets with realistic patterns.
- Includes additional columns such as `study_hours`, `parental_support`, `peer_influence`, and `stress_level`.
- Normalizes numerical data and one-hot encodes categorical data for GAN training.
- Saves the generated synthetic dataset as a CSV file for further analysis.

## Dataset

The synthetic dataset includes the following columns:

1. **Numerical Columns**:
   - `duration`: Time spent on activities (normalized).
   - `age`: Age of the individual (normalized).
   - `performance_score`: Performance score (normalized).
   - `study_hours`: Hours spent studying daily (normalized).
   - `stress_level`: Stress level on a scale of 1–10 (normalized).

2. **Categorical Columns** (One-Hot Encoded):
   - `activity_type`: Types of activities (e.g., Homework, Video Games, Reading, etc.).
   - `gender`: Gender (`F` or `M`).
   - `location`: Locations where activities are performed (e.g., Home, Library, Gym, etc.).
   - `interest_level`: Interest level (`Low`, `Medium`, `High`).
   - `parental_support`: Level of parental support (`Low`, `Medium`, `High`).
   - `peer_influence`: Influence from peers (`Negative`, `Neutral`, `Positive`).

## How It Works

1. **Data Preprocessing**:
   - The real dataset is normalized for numerical columns and one-hot encoded for categorical columns.
   - The processed data is used as input for the GAN.

2. **GAN Architecture**:
   - **Generator**: Creates synthetic data from random noise.
   - **Discriminator**: Distinguishes between real and synthetic data.
   - The GAN is trained iteratively to improve the Generator's ability to create realistic data.

3. **Synthetic Data Generation**:
   - After training, the Generator is used to create synthetic data.
   - The synthetic data is saved as `gan_synthetic_evening_activities_with_more_columns.csv`.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>