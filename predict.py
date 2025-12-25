"""
Smart Health Diagnosis - Prediction Script
This script allows users to input symptoms and get disease predictions.
"""

import joblib
import numpy as np
import os

# Disease precautions and recommendations
DISEASE_INFO = {
    "Influenza": {
        "precautions": ["Get plenty of rest", "Stay hydrated", "Take antiviral medications if prescribed"],
        "severity": "Moderate"
    },
    "Common Cold": {
        "precautions": ["Rest well", "Drink warm fluids", "Use over-the-counter cold medicine"],
        "severity": "Mild"
    },
    "Dengue Fever": {
        "precautions": ["Seek immediate medical attention", "Stay hydrated", "Monitor platelet count"],
        "severity": "Severe"
    },
    "Malaria": {
        "precautions": ["Take antimalarial medication", "Seek immediate medical care", "Rest completely"],
        "severity": "Severe"
    },
    "Typhoid": {
        "precautions": ["Take prescribed antibiotics", "Maintain hygiene", "Eat easily digestible food"],
        "severity": "Severe"
    },
    "Gastroenteritis": {
        "precautions": ["Stay hydrated with ORS", "Eat light bland food", "Rest and avoid dairy"],
        "severity": "Moderate"
    },
    "Pneumonia": {
        "precautions": ["Take prescribed antibiotics", "Get plenty of rest", "Use humidifier"],
        "severity": "Severe"
    },
    "Bronchitis": {
        "precautions": ["Avoid smoking", "Use steam inhalation", "Take prescribed medications"],
        "severity": "Moderate"
    },
    "Anemia": {
        "precautions": ["Eat iron-rich foods", "Take iron supplements", "Consult a hematologist"],
        "severity": "Moderate"
    },
    "Arthritis": {
        "precautions": ["Regular gentle exercise", "Apply hot/cold therapy", "Maintain healthy weight"],
        "severity": "Chronic"
    },
    "Hepatitis A": {
        "precautions": ["Rest completely", "Avoid alcohol", "Maintain proper hygiene"],
        "severity": "Moderate"
    },
    "Hepatitis B": {
        "precautions": ["Take antiviral medications", "Avoid alcohol", "Regular liver function tests"],
        "severity": "Severe"
    },
    "Migraine": {
        "precautions": ["Rest in a dark quiet room", "Take prescribed painkillers", "Identify triggers"],
        "severity": "Moderate"
    },
    "Acid Reflux": {
        "precautions": ["Avoid spicy foods", "Eat smaller meals", "Don't lie down after eating"],
        "severity": "Mild"
    },
    "Chickenpox": {
        "precautions": ["Isolate from others", "Use calamine lotion", "Keep nails short"],
        "severity": "Moderate"
    },
    "COVID-19": {
        "precautions": ["Isolate immediately", "Monitor oxygen levels", "Seek medical care if severe"],
        "severity": "Variable"
    },
    "Psoriasis": {
        "precautions": ["Moisturize regularly", "Avoid triggers", "Use prescribed topical treatments"],
        "severity": "Chronic"
    },
    "Asthma": {
        "precautions": ["Use inhaler as prescribed", "Avoid triggers", "Regular breathing exercises"],
        "severity": "Chronic"
    },
    "Tuberculosis": {
        "precautions": ["Complete full course of antibiotics", "Isolate during treatment", "Regular check-ups"],
        "severity": "Severe"
    },
    "Diabetes": {
        "precautions": ["Monitor blood sugar", "Follow diabetic diet", "Regular exercise"],
        "severity": "Chronic"
    },
    "Hypertension": {
        "precautions": ["Reduce salt intake", "Regular exercise", "Monitor blood pressure"],
        "severity": "Chronic"
    },
    "Allergies": {
        "precautions": ["Avoid allergens", "Take antihistamines", "Keep environment clean"],
        "severity": "Mild"
    },
    "Food Poisoning": {
        "precautions": ["Stay hydrated", "Rest completely", "Seek medical help if severe"],
        "severity": "Moderate"
    },
    "Tonsillitis": {
        "precautions": ["Gargle with warm salt water", "Rest your voice", "Take prescribed antibiotics"],
        "severity": "Moderate"
    },
    "Strep Throat": {
        "precautions": ["Complete antibiotic course", "Rest your voice", "Drink warm liquids"],
        "severity": "Moderate"
    },
    "Chikungunya": {
        "precautions": ["Rest completely", "Stay hydrated", "Take pain relievers"],
        "severity": "Moderate"
    },
    "HIV/AIDS": {
        "precautions": ["Take antiretroviral therapy", "Regular medical check-ups", "Healthy lifestyle"],
        "severity": "Severe"
    },
    "Eczema": {
        "precautions": ["Moisturize regularly", "Avoid harsh soaps", "Wear soft fabrics"],
        "severity": "Chronic"
    },
    "Tension Headache": {
        "precautions": ["Manage stress", "Get adequate sleep", "Regular breaks from screens"],
        "severity": "Mild"
    },
    "Peptic Ulcer": {
        "precautions": ["Avoid spicy and acidic foods", "Take prescribed medications", "Reduce stress"],
        "severity": "Moderate"
    },
    "Urinary Tract Infection": {
        "precautions": ["Drink plenty of water", "Take prescribed antibiotics", "Maintain hygiene"],
        "severity": "Moderate"
    },
    "Anxiety Disorder": {
        "precautions": ["Practice relaxation techniques", "Seek therapy", "Regular exercise"],
        "severity": "Chronic"
    },
    "Depression": {
        "precautions": ["Seek professional help", "Maintain social connections", "Regular physical activity"],
        "severity": "Chronic"
    },
    "Sinusitis": {
        "precautions": ["Steam inhalation", "Use saline nasal spray", "Stay hydrated"],
        "severity": "Mild"
    }
}

def load_model():
    """Load the trained model and feature names."""
    try:
        model = joblib.load('models/disease_model.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, feature_names
    except FileNotFoundError:
        print("❌ Error: Model not found!")
        print("   Please run 'python train_model.py' first to train the model.")
        return None, None

def get_symptoms_input(feature_names):
    """Get symptom input from user."""
    print("\n" + "=" * 60)
    print(" SMART HEALTH DIAGNOSIS SYSTEM")
    print("=" * 60)
    print("\nPlease answer the following questions about your symptoms.")
    print("Enter 1 for YES or 0 for NO\n")
    print("-" * 60)
    
    symptoms = {}
    symptom_questions = {
        "Fever": "Do you have a fever (elevated body temperature)?",
        "Cough": "Do you have a cough?",
        "Fatigue": "Are you feeling unusually tired or fatigued?",
        "Headache": "Do you have a headache?",
        "Nausea": "Are you feeling nauseous?",
        "Vomiting": "Have you been vomiting?",
        "Diarrhea": "Do you have diarrhea?",
        "Muscle_Pain": "Do you have muscle pain or body aches?",
        "Joint_Pain": "Do you have joint pain?",
        "Skin_Rash": "Do you have any skin rash?",
        "Runny_Nose": "Do you have a runny or stuffy nose?",
        "Sore_Throat": "Do you have a sore throat?",
        "Chest_Pain": "Do you have chest pain or discomfort?",
        "Shortness_of_Breath": "Are you experiencing shortness of breath?",
        "Abdominal_Pain": "Do you have abdominal pain?",
        "Loss_of_Appetite": "Have you lost your appetite?",
        "Dizziness": "Are you feeling dizzy?",
        "Chills": "Do you have chills?",
        "Sweating": "Are you experiencing excessive sweating?",
        "Weight_Loss": "Have you experienced unexplained weight loss?"
    }
    
    for feature in feature_names:
        question = symptom_questions.get(feature, f"Do you have {feature.replace('_', ' ').lower()}?")
        while True:
            try:
                response = input(f"  {question} (1/0): ").strip()
                if response in ['0', '1']:
                    symptoms[feature] = int(response)
                    break
                else:
                    print("     Please enter 1 for YES or 0 for NO")
            except KeyboardInterrupt:
                print("\n\n Goodbye!")
                exit()
    
    return symptoms

def predict_disease(model, feature_names, symptoms):
    """Make a prediction based on symptoms."""
    # Create feature vector
    features = np.array([[symptoms[f] for f in feature_names]])
    
    # Get prediction and probability
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Get confidence score
    max_prob_idx = np.argmax(probabilities)
    confidence = probabilities[max_prob_idx] * 100
    
    # Get top 3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_diseases = [(model.classes_[i], probabilities[i] * 100) for i in top_indices]
    
    return prediction, confidence, top_diseases

def display_results(prediction, confidence, top_diseases):
    """Display the prediction results."""
    print("\n" + "=" * 60)
    print(" DIAGNOSIS RESULTS")
    print("=" * 60)
    
    # Display main prediction
    print(f"\n Primary Diagnosis: {prediction}")
    print(f" Confidence: {confidence:.1f}%")
    
    # Get disease info
    info = DISEASE_INFO.get(prediction, {
        "precautions": ["Consult a healthcare professional"],
        "severity": "Unknown"
    })
    
    # Display severity
    severity_icons = {
        "Mild": "🟢",
        "Moderate": "🟡",
        "Severe": "🔴",
        "Chronic": "🔵",
        "Variable": "⚪",
        "Unknown": "⚫"
    }
    icon = severity_icons.get(info["severity"], "⚫")
    print(f" Severity Level: {icon} {info['severity']}")
    
    # Display precautions
    print("\n Recommended Precautions:")
    for i, precaution in enumerate(info["precautions"], 1):
        print(f"   {i}. {precaution}")
    
    # Display other possibilities
    print("\n Other Possible Conditions:")
    for disease, prob in top_diseases[1:]:
        if prob > 5:  # Only show if probability > 5%
            print(f"   • {disease}: {prob:.1f}%")
    
    # Disclaimer
    print("\n" + "-" * 60)
    print(" IMPORTANT DISCLAIMER:")
    print("   This is an AI-based preliminary assessment only.")
    print("   Please consult a qualified healthcare professional")
    print("   for accurate diagnosis and treatment.")
    print("=" * 60)

def main():
    """Main function to run the prediction."""
    # Load model
    model, feature_names = load_model()
    if model is None:
        return
    
    while True:
        # Get symptoms
        symptoms = get_symptoms_input(feature_names)
        
        # Make prediction
        prediction, confidence, top_diseases = predict_disease(model, feature_names, symptoms)
        
        # Display results
        display_results(prediction, confidence, top_diseases)
        
        # Ask if user wants to continue
        print("\n")
        again = input("Would you like to check another diagnosis? (y/n): ").strip().lower()
        if again != 'y':
            print("\n Thank you for using Smart Health Diagnosis. Stay healthy!")
            break

if __name__ == "__main__":
    main()
