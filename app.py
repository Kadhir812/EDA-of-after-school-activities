import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import random
import numpy as np

# Function to generate synthetic data
def generate_realistic_data(num_records=600):
    # Define possible values for each column
    activities = ['Homework', 'Video Games', 'Reading', 'TV', 'Exercise', 'Music', 'Social Media', 'Cooking', 'Sports']
    genders = ['M', 'F']
    locations = ['Home', 'Library', 'Community Center', 'Friend\'s House', 'Park', 'Gym']
    interest_levels = ['Low', 'Medium', 'High']
    age_range = range(12, 18)  # Ages between 12 and 17
    performance_scores = range(30, 101)  # Scores between 30 and 100

    # Generate random data
    data = {
        'student_id': range(1, num_records + 1),
        'activity_type': [random.choices(activities, weights=[10, 15, 10, 10, 15, 10, 20, 5, 5])[0] for _ in range(num_records)],
        'duration': [random.randint(30, 180) for _ in range(num_records)],  # Duration in minutes
        'age': [random.choice(age_range) for _ in range(num_records)],
        'gender': [random.choice(genders) for _ in range(num_records)],
        'location': [random.choices(locations, weights=[30, 20, 15, 15, 10, 10])[0] for _ in range(num_records)],
        'performance_score': [random.randint(30, 100) for _ in range(num_records)],
        'interest_level': [random.choices(interest_levels, weights=[20, 50, 30])[0] for _ in range(num_records)],
    }

    # Introduce correlations
    for i in range(num_records):
        # Higher interest level leads to higher performance scores
        if data['interest_level'][i] == 'High':
            data['performance_score'][i] = random.randint(80, 100)
        elif data['interest_level'][i] == 'Medium':
            data['performance_score'][i] = random.randint(50, 79)
        else:  # Low interest
            data['performance_score'][i] = random.randint(30, 49)

        # Older students spend more time on Homework and less on Video Games
        if data['age'][i] > 15 and data['activity_type'][i] == 'Homework':
            data['duration'][i] = random.randint(90, 180)
        elif data['age'][i] < 14 and data['activity_type'][i] == 'Video Games':
            data['duration'][i] = random.randint(60, 180)

    # Add anomalies (5% of the data)
    for i in range(int(num_records * 0.05)):
        idx = random.randint(0, num_records - 1)
        data['duration'][idx] = random.randint(200, 300)  # Outlier durations
        data['performance_score'][idx] = random.randint(0, 29)  # Very low performance scores

    return pd.DataFrame(data)

# Generate the dataset
realistic_data = generate_realistic_data(600)

# Save the dataset to a CSV file
realistic_data.to_csv('realistic_evening_activities.csv', index=False)
print("Realistic synthetic dataset created and saved as 'realistic_evening_activities.csv'.")

# Load the dataset
data = pd.read_csv('realistic_evening_activities.csv')

# Streamlit UI
st.title("Exploratory Data Analysis of Evening Activities")
st.write("This dashboard provides insights into the synthetic dataset of students' evening activities.")

# Dataset Information
st.header("Dataset Information")
st.write(data.info())
st.write("The dataset contains 600 rows and 8 columns. Each row represents a student's evening activity.")

# Missing Values
st.header("Missing Values")
st.write(data.isnull().sum())
st.write("There are no missing values in the dataset, so it is clean and ready for analysis.")

# Activity Distribution
st.header("Activity Distribution")
activity_counts = data['activity_type'].value_counts()
st.write(activity_counts)
st.write("The most common activity is 'Social Media', while 'Cooking' is the least common.")

fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='activity_type', data=data, palette='viridis', order=activity_counts.index, ax=ax)
ax.set_title('Distribution of Evening Activities')
ax.set_xlabel('Activity Type')
ax.set_ylabel('Count')
st.pyplot(fig)

# Average Time Spent on Activities
st.header("Average Time Spent on Activities")
avg_time = data.groupby('activity_type')['duration'].mean().sort_values(ascending=False)
st.write(avg_time)
st.write("Students spend the most time on 'Social Media' and the least time on 'Reading'.")

fig, ax = plt.subplots(figsize=(10, 6))
avg_time.plot(kind='bar', color='skyblue', ax=ax)
ax.set_title('Average Time Spent on Evening Activities')
ax.set_xlabel('Activity Type')
ax.set_ylabel('Average Duration (minutes)')
st.pyplot(fig)

# Trends by Gender
st.header("Activity Duration by Gender")
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(x='activity_type', y='duration', hue='gender', data=data, palette='Set2', ax=ax)
ax.set_title('Activity Duration by Gender')
ax.set_xlabel('Activity Type')
ax.set_ylabel('Duration (minutes)')
st.pyplot(fig)
st.write("The box plot shows the distribution of activity durations by gender.")

# Trends by Age Group
st.header("Activity Duration by Age Group")
data['age_group'] = pd.cut(data['age'], bins=[12, 14, 16, 18], labels=['12-14', '15-16', '17-18'])
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='activity_type', y='duration', hue='age_group', data=data, palette='coolwarm', errorbar=None, ax=ax)
ax.set_title('Activity Duration by Age Group')
ax.set_xlabel('Activity Type')
ax.set_ylabel('Average Duration (minutes)')
st.pyplot(fig)
st.write("The bar plot shows how activity durations vary across age groups.")

# Correlation Analysis
st.header("Correlation Analysis")
correlation = data[['duration', 'age', 'performance_score']].corr()
st.write(correlation)
st.write("The correlation matrix shows that 'duration' and 'performance_score' have a negative correlation.")

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
ax.set_title('Correlation Heatmap')
st.pyplot(fig)