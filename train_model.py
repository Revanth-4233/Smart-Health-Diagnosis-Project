"""
Smart Health Diagnosis - Model Training Script
This script trains a machine learning model to predict diseases based on symptoms.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def load_data():
    """Load and preprocess the symptoms dataset."""
    print("=" * 60)
    print(" SMART HEALTH DIAGNOSIS - MODEL TRAINING")
    print("=" * 60)
    print("\n Loading dataset...")
    
    df = pd.read_csv('data/symptoms.csv')
    print(f" Loaded {len(df)} patient records")
    print(f" Features: {df.columns.tolist()}")
    
    return df

def prepare_features(df):
    """Prepare features and target variable."""
    print("\n Preparing features...")
    
    # Separate features and target
    X = df.drop(['Symptoms', 'Disease'], axis=1)
    y = df['Disease']
    
    # Get unique diseases
    diseases = y.unique()
    print(f" Found {len(diseases)} unique diseases:")
    for disease in sorted(diseases):
        count = (y == disease).sum()
        print(f"   • {disease}: {count} samples")
    
    return X, y

def train_model(X, y):
    """Train and evaluate machine learning models."""
    print("\n Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35, random_state=42, stratify=y
    )

    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Train Decision Tree
    print("\n Training Decision Tree Classifier...")
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt_model.fit(X_train, y_train)
    dt_predictions = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_predictions)
    print(f"   Decision Tree Accuracy: {dt_accuracy:.2%}")
    
    # Train Random Forest
    print("\n Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    print(f"   Random Forest Accuracy: {rf_accuracy:.2%}")
    
    # Choose the best model
    if rf_accuracy >= dt_accuracy:
        best_model = rf_model
        best_name = "Random Forest"
        best_accuracy = rf_accuracy
        predictions = rf_predictions
    else:
        best_model = dt_model
        best_name = "Decision Tree"
        best_accuracy = dt_accuracy
        predictions = dt_predictions
    
    print(f"\n Best Model: {best_name} with {best_accuracy:.2%} accuracy")
    
    # Print classification report
    print("\n Classification Report:")
    print("-" * 60)
    print(classification_report(y_test, predictions))
    
    return best_model, X.columns.tolist()

def save_model(model, feature_names):
    """Save the trained model and feature names."""
    print("\n Saving model...")
    
    # Save the model
    model_path = 'models/disease_model.pkl'
    joblib.dump(model, model_path)
    print(f"   Model saved to: {model_path}")
    
    # Save feature names
    features_path = 'models/feature_names.pkl'
    joblib.dump(feature_names, features_path)
    print(f"   Features saved to: {features_path}")
    
    print("\n Model training complete!")
    print("=" * 60)

def main():
    """Main function to run the training pipeline."""
    try:
        # Load data
        df = load_data()
        
        # Prepare features
        X, y = prepare_features(df)
        
        # Train model
        model, feature_names = train_model(X, y)
        
        # Save model
        save_model(model, feature_names)
        
        print("\n You can now use 'python predict.py' to make predictions!")
        print(" Or run 'python app.py' to start the web interface!")
        
    except FileNotFoundError:
        print(" Error: 'data/symptoms.csv' not found!")
        print("   Please ensure the data file exists in the 'data' directory.")
    except Exception as e:
        print(f" Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
