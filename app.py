"""
Smart Health Diagnosis - Flask Web Application
A web-based interface for symptom-based disease prediction.
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Disease information database
DISEASE_INFO = {
    "Influenza": {"precautions": ["Get plenty of rest", "Stay hydrated", "Take antiviral medications if prescribed"], "severity": "Moderate", "color": "#FFA500"},
    "Common Cold": {"precautions": ["Rest well", "Drink warm fluids", "Use over-the-counter cold medicine"], "severity": "Mild", "color": "#4CAF50"},
    "Dengue Fever": {"precautions": ["Seek immediate medical attention", "Stay hydrated", "Monitor platelet count"], "severity": "Severe", "color": "#F44336"},
    "Malaria": {"precautions": ["Take antimalarial medication", "Seek immediate medical care", "Rest completely"], "severity": "Severe", "color": "#F44336"},
    "Typhoid": {"precautions": ["Take prescribed antibiotics", "Maintain hygiene", "Eat easily digestible food"], "severity": "Severe", "color": "#F44336"},
    "Gastroenteritis": {"precautions": ["Stay hydrated with ORS", "Eat light bland food", "Rest and avoid dairy"], "severity": "Moderate", "color": "#FFA500"},
    "Pneumonia": {"precautions": ["Take prescribed antibiotics", "Get plenty of rest", "Use humidifier"], "severity": "Severe", "color": "#F44336"},
    "Bronchitis": {"precautions": ["Avoid smoking", "Use steam inhalation", "Take prescribed medications"], "severity": "Moderate", "color": "#FFA500"},
    "Anemia": {"precautions": ["Eat iron-rich foods", "Take iron supplements", "Consult a hematologist"], "severity": "Moderate", "color": "#FFA500"},
    "Arthritis": {"precautions": ["Regular gentle exercise", "Apply hot/cold therapy", "Maintain healthy weight"], "severity": "Chronic", "color": "#2196F3"},
    "Hepatitis A": {"precautions": ["Rest completely", "Avoid alcohol", "Maintain proper hygiene"], "severity": "Moderate", "color": "#FFA500"},
    "Hepatitis B": {"precautions": ["Take antiviral medications", "Avoid alcohol", "Regular liver function tests"], "severity": "Severe", "color": "#F44336"},
    "Migraine": {"precautions": ["Rest in a dark quiet room", "Take prescribed painkillers", "Identify triggers"], "severity": "Moderate", "color": "#FFA500"},
    "Acid Reflux": {"precautions": ["Avoid spicy foods", "Eat smaller meals", "Don't lie down after eating"], "severity": "Mild", "color": "#4CAF50"},
    "Chickenpox": {"precautions": ["Isolate from others", "Use calamine lotion", "Keep nails short"], "severity": "Moderate", "color": "#FFA500"},
    "COVID-19": {"precautions": ["Isolate immediately", "Monitor oxygen levels", "Seek medical care if severe"], "severity": "Variable", "color": "#9C27B0"},
    "Psoriasis": {"precautions": ["Moisturize regularly", "Avoid triggers", "Use prescribed topical treatments"], "severity": "Chronic", "color": "#2196F3"},
    "Asthma": {"precautions": ["Use inhaler as prescribed", "Avoid triggers", "Regular breathing exercises"], "severity": "Chronic", "color": "#2196F3"},
    "Tuberculosis": {"precautions": ["Complete full course of antibiotics", "Isolate during treatment", "Regular check-ups"], "severity": "Severe", "color": "#F44336"},
    "Diabetes": {"precautions": ["Monitor blood sugar", "Follow diabetic diet", "Regular exercise"], "severity": "Chronic", "color": "#2196F3"},
    "Hypertension": {"precautions": ["Reduce salt intake", "Regular exercise", "Monitor blood pressure"], "severity": "Chronic", "color": "#2196F3"},
    "Allergies": {"precautions": ["Avoid allergens", "Take antihistamines", "Keep environment clean"], "severity": "Mild", "color": "#4CAF50"},
    "Food Poisoning": {"precautions": ["Stay hydrated", "Rest completely", "Seek medical help if severe"], "severity": "Moderate", "color": "#FFA500"},
    "Tonsillitis": {"precautions": ["Gargle with warm salt water", "Rest your voice", "Take prescribed antibiotics"], "severity": "Moderate", "color": "#FFA500"},
    "Strep Throat": {"precautions": ["Complete antibiotic course", "Rest your voice", "Drink warm liquids"], "severity": "Moderate", "color": "#FFA500"},
    "Chikungunya": {"precautions": ["Rest completely", "Stay hydrated", "Take pain relievers"], "severity": "Moderate", "color": "#FFA500"},
    "HIV/AIDS": {"precautions": ["Take antiretroviral therapy", "Regular medical check-ups", "Healthy lifestyle"], "severity": "Severe", "color": "#F44336"},
    "Eczema": {"precautions": ["Moisturize regularly", "Avoid harsh soaps", "Wear soft fabrics"], "severity": "Chronic", "color": "#2196F3"},
    "Tension Headache": {"precautions": ["Manage stress", "Get adequate sleep", "Regular breaks from screens"], "severity": "Mild", "color": "#4CAF50"},
    "Peptic Ulcer": {"precautions": ["Avoid spicy and acidic foods", "Take prescribed medications", "Reduce stress"], "severity": "Moderate", "color": "#FFA500"},
    "Urinary Tract Infection": {"precautions": ["Drink plenty of water", "Take prescribed antibiotics", "Maintain hygiene"], "severity": "Moderate", "color": "#FFA500"},
    "Anxiety Disorder": {"precautions": ["Practice relaxation techniques", "Seek therapy", "Regular exercise"], "severity": "Chronic", "color": "#2196F3"},
    "Depression": {"precautions": ["Seek professional help", "Maintain social connections", "Regular physical activity"], "severity": "Chronic", "color": "#2196F3"},
    "Sinusitis": {"precautions": ["Steam inhalation", "Use saline nasal spray", "Stay hydrated"], "severity": "Mild", "color": "#4CAF50"}
}

# Symptom information for the form
SYMPTOMS = [
    {"id": "Fever", "name": "Fever", "description": "Elevated body temperature"},
    {"id": "Cough", "name": "Cough", "description": "Persistent coughing"},
    {"id": "Fatigue", "name": "Fatigue", "description": "Unusual tiredness or exhaustion"},
    {"id": "Headache", "name": "Headache", "description": "Pain in the head"},
    {"id": "Nausea", "name": "Nausea", "description": "Feeling of sickness"},
    {"id": "Vomiting", "name": "Vomiting", "description": "Throwing up"},
    {"id": "Diarrhea", "name": "Diarrhea", "description": "Loose or watery stools"},
    {"id": "Muscle_Pain", "name": "Muscle Pain", "description": "Body aches and muscle soreness"},
    {"id": "Joint_Pain", "name": "Joint Pain", "description": "Pain in joints"},
    {"id": "Skin_Rash", "name": "Skin Rash", "description": "Skin irritation or redness"},
    {"id": "Runny_Nose", "name": "Runny Nose", "description": "Nasal discharge"},
    {"id": "Sore_Throat", "name": "Sore Throat", "description": "Pain or irritation in throat"},
    {"id": "Chest_Pain", "name": "Chest Pain", "description": "Pain or discomfort in chest"},
    {"id": "Shortness_of_Breath", "name": "Shortness of Breath", "description": "Difficulty breathing"},
    {"id": "Abdominal_Pain", "name": "Abdominal Pain", "description": "Pain in stomach area"},
    {"id": "Loss_of_Appetite", "name": "Loss of Appetite", "description": "Reduced desire to eat"},
    {"id": "Dizziness", "name": "Dizziness", "description": "Feeling lightheaded or unsteady"},
    {"id": "Chills", "name": "Chills", "description": "Feeling cold and shivering"},
    {"id": "Sweating", "name": "Sweating", "description": "Excessive perspiration"},
    {"id": "Weight_Loss", "name": "Weight Loss", "description": "Unexplained weight reduction"}
]

def load_model():
    """Load the trained model and feature names."""
    try:
        model = joblib.load('models/disease_model.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, feature_names
    except FileNotFoundError:
        return None, None

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html', symptoms=SYMPTOMS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    model, feature_names = load_model()
    
    if model is None:
        return jsonify({
            'error': True,
            'message': 'Model not found. Please run train_model.py first.'
        })
    
    try:
        # Get symptoms from form
        symptoms = {}
        for feature in feature_names:
            symptoms[feature] = int(request.form.get(feature, 0))
        
        # Create feature vector
        features = np.array([[symptoms[f] for f in feature_names]])
        
        # Get prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get confidence score
        max_prob_idx = np.argmax(probabilities)
        confidence = float(probabilities[max_prob_idx] * 100)
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        other_predictions = []
        for i in top_indices[1:]:
            prob = probabilities[i] * 100
            if prob > 5:
                other_predictions.append({
                    'disease': model.classes_[i],
                    'probability': round(prob, 1)
                })
        
        # Get disease info
        info = DISEASE_INFO.get(prediction, {
            "precautions": ["Consult a healthcare professional"],
            "severity": "Unknown",
            "color": "#9E9E9E"
        })
        
        return jsonify({
            'error': False,
            'prediction': prediction,
            'confidence': round(confidence, 1),
            'severity': info['severity'],
            'color': info['color'],
            'precautions': info['precautions'],
            'other_predictions': other_predictions
        })
        
    except Exception as e:
        return jsonify({
            'error': True,
            'message': str(e)
        })

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(" SMART HEALTH DIAGNOSIS - WEB APPLICATION")
    print("=" * 60)
    print("\n Starting web server...")
    print(" Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server.\n")
    app.run(debug=True, port=5000)
