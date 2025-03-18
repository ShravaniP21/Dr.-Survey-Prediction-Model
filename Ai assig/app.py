import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template, send_file

app = Flask(__name__)

# Load trained model
model = joblib.load("survey_prediction_model.pkl")

# Load dataset (to map predictions back to NPIs)
df = pd.read_csv("npi dataset.csv")

# Convert timestamps to datetime
df['Login Time'] = pd.to_datetime(df['Login Time'])

# Function to get the best doctors for a given time
def get_best_doctors(hour):
    # Load the original dataset
    df_original = pd.read_csv("npi dataset.csv")

    # Convert timestamps to datetime
    df_original['Login Time'] = pd.to_datetime(df_original['Login Time'])
    df_original['Logout Time'] = pd.to_datetime(df_original['Logout Time'])

    # Feature Engineering (same as training)
    df_original['Session Duration'] = (df_original['Logout Time'] - df_original['Login Time']).dt.total_seconds() / 60
    df_original['Login Hour'] = df_original['Login Time'].dt.hour
    df_original['Login Day'] = df_original['Login Time'].dt.day
    df_original['Login Month'] = df_original['Login Time'].dt.month
    df_original['Login Weekday'] = df_original['Login Time'].dt.weekday

    df_original['Survey Participation'] = (df_original['Count of Survey Attempts'] > 0).astype(int)

    # One-hot encode categorical features
    df_original = pd.get_dummies(df_original, columns=['State', 'Region'], drop_first=True)

    # Label encode 'Speciality'
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df_original['Speciality'] = le.fit_transform(df_original['Speciality'])

    # Ensure all necessary features exist
    required_features = [
        'Login Hour', 'Login Day', 'Login Month', 'Login Weekday', 'Session Duration',
        'Speciality', 'Count of Survey Attempts', 'State_FL', 'State_GA', 'State_IL',
        'State_MI', 'State_NC', 'State_NY', 'State_OH', 'State_PA', 'State_TX',
        'Region_Northeast', 'Region_South', 'Region_West'
    ]
    
    # Ensure all missing columns are added with default value 0
    for feature in required_features:
        if feature not in df_original.columns:
            df_original[feature] = 0  # Default to 0 for missing columns

    # Filter dataset to only include doctors available at the given hour
    available_doctors = df_original[df_original['Login Hour'] == hour]

    # Select only the relevant features for prediction
    X_test = available_doctors[required_features]

    # Predict probabilities using the trained model
    probabilities = model.predict_proba(X_test)[:, 1]  # Probability of participation

    # Add probabilities to the dataset
    available_doctors['Participation Probability'] = probabilities

    # Sort doctors by highest probability
    best_doctors = available_doctors.sort_values(by='Participation Probability', ascending=False)[['NPI', 'Participation Probability']]

    # Save results as CSV
    best_doctors.to_csv("best_doctors.csv", index=False, float_format="%.10f")

    return best_doctors

# Route: Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Route: Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get time input from the user
        hour = int(request.form['hour'])

        # Get best doctors
        best_doctors = get_best_doctors(hour)

        return send_file("best_doctors.csv", as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
