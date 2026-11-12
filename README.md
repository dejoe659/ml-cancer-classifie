# Breast Cancer Classification with Machine Learning

A machine learning project that classifies breast tumors as **malignant** or **benign** using scikit-learn. It trains several models, compares them with cross-validation, and saves the best-performing one for future predictions.

## Overview

Early and accurate diagnosis of breast cancer is critical. This project uses the Breast Cancer Wisconsin dataset (built into scikit-learn) to build a classifier based on 30 numeric features computed from images of cell nuclei, such as radius, texture, perimeter, and smoothness.

## Features

- Loads and explores the dataset (569 samples, 30 features)
- Splits data into training and test sets with stratification
- Trains and compares four models:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Random Forest
- Evaluates models using 5-fold cross-validation
- Reports accuracy, precision, recall, and F1-score
- Visualizes results with a confusion matrix and model comparison chart
- Saves the best model with `joblib` and demonstrates loading it to predict

## Tech Stack

- Python 3
- scikit-learn
- pandas
- matplotlib
- joblib

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/ml-cancer-classifier.git
cd ml-cancer-classifier
pip install -r requirements.txt
```

## Usage

```bash
python ml_project.py
```

The script will print model comparison results and evaluation metrics, then generate:

- `results.png`: confusion matrix and model comparison chart
- `best_model.joblib`: the trained model, ready to reuse

## Example: Using the Saved Model

```python
import joblib

model = joblib.load("best_model.joblib")
prediction = model.predict(new_data)  # 0 = malignant, 1 = benign
```

## Results

The models typically reach around **95-98% accuracy** on the test set. Exact results may vary slightly depending on the best-performing model.

## Future Improvements

- Hyperparameter tuning with GridSearchCV
- Try deep learning models (TensorFlow or PyTorch)
- Build a simple web app for predictions using Streamlit or Flask
- Add more evaluation metrics such as ROC curve and AUC

## Disclaimer

This project is for educational purposes only and should not be used for real medical diagnosis.

## License

MIT License
