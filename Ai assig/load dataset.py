import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("npi dataset.csv")

# Convert Login and Logout Time to datetime format
df['Login Time'] = pd.to_datetime(df['Login Time'])
df['Logout Time'] = pd.to_datetime(df['Logout Time'])

# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Calculate session duration in minutes
df['Session Duration'] = (df['Logout Time'] - df['Login Time']).dt.total_seconds() / 60

# Extract features from datetime columns
df['Login Hour'] = df['Login Time'].dt.hour
df['Login Day'] = df['Login Time'].dt.day
df['Login Month'] = df['Login Time'].dt.month
df['Login Weekday'] = df['Login Time'].dt.weekday  # Monday=0, Sunday=6

# Categorize the time of day
def get_time_category(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['Time of Day'] = df['Login Hour'].apply(get_time_category)

# Encode categorical features
df = pd.get_dummies(df, columns=['State', 'Region'], drop_first=True)

# Apply Label Encoding to Speciality
le = LabelEncoder()
df['Speciality'] = le.fit_transform(df['Speciality'])

# Strip leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Display dataset info
print("\nDataset Info:")
df.info()

# Display first few rows
print("\nFirst 5 rows:")
print(df.head())

# Print column names to verify
print("\nColumn Names:")
print(df.columns)
