"""
Model training functions that can be imported by Streamlit app
"""
import pickle
import os
from pathlib import Path
from train_model import create_and_train_model

def ensure_model_exists(model_path='models/resume_classifier.pkl'):
    """
    Ensures the model exists. If not, creates and trains it.
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        # Create models directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Train and save the model
        model = create_and_train_model()
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return True  # Model was trained
    
    return False  # Model already existed
