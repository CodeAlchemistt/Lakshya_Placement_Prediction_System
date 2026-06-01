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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Grab all the input values from the HTML form
        input_data = [float(request.form.get(feature, 0)) for feature in FEATURES]
        
        # 2. Convert to a numpy array and reshape it for the model
        input_array = np.array(input_data).reshape(1, -1)
        
        # 3. Make the prediction!
        prediction = model.predict(input_array)[0]
        
        # 4. Interpret the result (Assuming 1 = Placed, 0 = Not Placed)
        result = "Placed" if prediction == 1 else "Not Placed"
        
        return render_template('index.html', features=FEATURES, prediction_text=f'AI Prediction: Student will be {result}')
    
    except Exception as e:
        return render_template('index.html', features=FEATURES, prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    # Run the app locally for testing
    app.run(debug=True)