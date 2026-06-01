import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import os

# Define file paths
CSV_PATH = '../data/student_placement_prediction_dataset_2026.csv'
MODEL_PATH = 'placement_model.pkl'

def train_model():
    print("Loading dataset...")
    try:
        df = pd.read_csv(CSV_PATH)
        print("Columns in CSV:", df.columns.tolist())
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {CSV_PATH}")
        return

    try:
        df = pd.read_csv(CSV_PATH)
        print("Columns in CSV:", df.columns.tolist())
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {CSV_PATH}")
        return

    # --- ADD THIS NEW BLOCK OF CODE ---
    print("Translating text data into numbers...")
    encoder = LabelEncoder()
    # Loop through every column
    for col in df.columns:
        # If the column contains text/objects
        if df[col].dtype == 'object': 
            df[col] = encoder.fit_transform(df[col])
        
    FEATURES = [
      'age', 'cgpa', 'internships_count', 'projects_count',
    'certifications_count', 'coding_skill_score', 'aptitude_score', 
    'communication_skill_score', 'logical_reasoning_score', 
    'hackathons_participated', 'github_repos', 'linkedin_connections', 
    'mock_interview_score', 'attendance_percentage', 'backlogs', 
    'extracurricular_score', 'leadership_score', 'volunteer_experience', 
    'study_hours_per_day']

    TARGET = 'placement_status'

    # Separate data into X (features) and y (target)
    X = df[FEATURES]
    y = df[TARGET]

    # Split into 80% training data and 20% testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Test the model's accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model trained successfully! Accuracy: {accuracy * 100:.2f}%")

    # Save the model to a .pkl file
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()