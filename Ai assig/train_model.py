import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load preprocessed dataset
df = pd.read_csv("npi dataset.csv")

# Convert Login and Logout Time to datetime format
df['Login Time'] = pd.to_datetime(df['Login Time'])
df['Logout Time'] = pd.to_datetime(df['Logout Time'])

# Calculate session duration
df['Session Duration'] = (df['Logout Time'] - df['Login Time']).dt.total_seconds() / 60

# Extract time-based features
df['Login Hour'] = df['Login Time'].dt.hour
df['Login Day'] = df['Login Time'].dt.day
df['Login Month'] = df['Login Time'].dt.month
df['Login Weekday'] = df['Login Time'].dt.weekday  # Monday=0, Sunday=6

# Define time category
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

# Encode categorical variables
df = pd.get_dummies(df, columns=['State', 'Region'], drop_first=True)

# Label encode 'Speciality'
le = LabelEncoder()
df['Speciality'] = le.fit_transform(df['Speciality'])

# Define features (X) and target (y)
features = [
    'Login Hour', 'Login Day', 'Login Month', 'Login Weekday', 'Session Duration',
    'Speciality', 'Count of Survey Attempts', 'State_FL', 'State_GA', 'State_IL',
    'State_MI', 'State_NC', 'State_NY', 'State_OH', 'State_PA', 'State_TX',
    'Region_Northeast', 'Region_South', 'Region_West'
]
df['Survey Participation'] = (df['Count of Survey Attempts'] > 0).astype(int)
target = 'Survey Participation'

X = df[features]
y = df[target]

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Print classification report
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "survey_prediction_model.pkl")
print("Model saved as 'survey_prediction_model.pkl'")
