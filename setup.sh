#!/bin/bash
if [ ! -f models/resume_classifier.pkl ]; then
    echo "Training model..."
    python train_model.py
else
    echo "Model already exists"
fi
