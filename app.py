from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
MODEL_PATH = 'ml_model/placement_model.pkl'
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

    FEATURES = [
      'age', 'cgpa', 'internships_count', 'projects_count',
    'certifications_count', 'coding_skill_score', 'aptitude_score', 
    'communication_skill_score', 'logical_reasoning_score', 
    'hackathons_participated', 'github_repos', 'linkedin_connections', 
    'mock_interview_score', 'attendance_percentage', 'backlogs', 
    'extracurricular_score', 'leadership_score', 'volunteer_experience', 
    'study_hours_per_day']

@app.route('/')
def home():
    # Pass the features list to our HTML so it can build the form automatically
    return render_template('index.html', features=FEATURES)

# The Landing Page
@app.route('/')
def index():
    return render_template('index.html')

# The Prediction Engine Page
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', features=FEATURES)

# The Analytics Page
@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

# The About Page
@app.route('/about')
def about():
    return render_template('about.html')

# The Prediction Logic (Triggered by the form on dashboard.html)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form.get(feature, 0)) for feature in FEATURES]
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        result = "Placed" if prediction == 1 else "Not Placed"
        
        # Notice we render dashboard.html here, not index.html!
        return render_template('dashboard.html', features=FEATURES, prediction_text=f'AI Prediction: Student will be {result}')
    
    except Exception as e:
        return render_template('dashboard.html', features=FEATURES, prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    # Run the app locally for testing
    app.run(debug=True)