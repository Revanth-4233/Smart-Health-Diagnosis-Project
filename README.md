# Smart Health Diagnosis System 🏥

An AI-powered health diagnosis system that predicts potential diseases based on user-reported symptoms using Machine Learning.

## Features

- **ML-Powered Diagnosis**: Uses Random Forest classifier for accurate disease prediction
- **30+ Diseases**: Trained on data covering common health conditions
- **20 Symptoms**: Comprehensive symptom selection for better accuracy
- **Web Interface**: Beautiful, responsive Flask web application
- **CLI Tool**: Command-line interface for quick diagnosis
- **Confidence Scores**: Shows prediction confidence and alternative diagnoses
- **Health Precautions**: Provides recommended precautions for each condition

## Project Structure

```
smart-health-diagnosis/
│
├── data/
│   └── symptoms.csv          # Symptom-disease dataset (120 records)
├── models/
│   ├── disease_model.pkl     # Trained ML model (generated)
│   └── feature_names.pkl     # Feature names (generated)
├── templates/
│   ├── index.html            # Web interface
│   └── about.html            # About page
├── train_model.py            # Model training script
├── predict.py                # CLI prediction tool
├── app.py                    # Flask web application
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. **Clone or download the project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**:
   ```bash
   python train_model.py
   ```

4. **Run predictions**:
   - **Web Interface**:
     ```bash
     python app.py
     ```
     Then open http://localhost:5000 in your browser.
   
   - **Command Line**:
     ```bash
     python predict.py
     ```

## Usage

### Web Interface
1. Open http://localhost:5000
2. Select your symptoms from the list
3. Click "Get Diagnosis"
4. View your diagnosis, confidence score, and precautions

### Command Line
1. Run `python predict.py`
2. Answer Yes/No (1/0) for each symptom
3. Get your diagnosis with recommendations

## Diseases Covered

The system can predict 30+ conditions including:
- Common Cold, Influenza, COVID-19
- Dengue Fever, Malaria, Typhoid
- Pneumonia, Bronchitis, Asthma
- Diabetes, Hypertension, Anemia
- And many more...

## Disclaimer

⚠️ **Important**: This is an AI-based preliminary assessment tool for educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for accurate diagnosis.

## Technologies Used

- Python 3.x
- Scikit-learn (Random Forest Classifier)
- Flask (Web Framework)
- Pandas & NumPy (Data Processing)
- HTML/CSS/JavaScript (Frontend)

## License

This project is for educational purposes.
