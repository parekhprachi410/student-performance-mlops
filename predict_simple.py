import pandas as pd
import numpy as np
import joblib
import sys

def predict_student_performance(student_data):
    """Make prediction for a single student"""
    
    # Load model and preprocessor
    model = joblib.load('models/artifacts/simple_model.pkl')
    preprocessor = joblib.load('models/artifacts/preprocessor.pkl')
    
    # Convert to DataFrame
    df = pd.DataFrame([student_data])
    
    # Create features (same as in training)
    df['average_score'] = (df['math_score'] + df['science_score'] + df['english_score']) / 3
    df['total_score'] = df['math_score'] + df['science_score'] + df['english_score']
    df['study_efficiency'] = df['average_score'] / (df['study_hours'] + 1)
    df['attendance_score'] = df['attendance_percentage'] / 100
    
    # Encode categorical variables
    for col in preprocessor['categorical_cols']:
        if col in df.columns:
            df[col] = preprocessor['label_encoders'][col].transform(df[col].astype(str))
    
    # Prepare features
    feature_cols = preprocessor['categorical_cols'] + preprocessor['numerical_cols'] + \
                   ['average_score', 'total_score', 'study_efficiency', 'attendance_score']
    
    X = preprocessor['scaler'].transform(df[feature_cols])
    
    # Predict
    pred_class = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    
    # Get grade names
    grade_names = preprocessor['label_encoders']['final_grade'].classes_
    predicted_grade = grade_names[pred_class]
    
    return predicted_grade, pred_proba

# Example student
example_student = {
    'age': 16,
    'gender': 'female',
    'school_type': 'private',
    'parent_education': 'graduate',
    'study_hours': 5.5,
    'attendance_percentage': 85.0,
    'internet_access': 'yes',
    'travel_time': '15-30 min',
    'extra_activities': 'yes',
    'study_method': 'notes',
    'math_score': 75.0,
    'science_score': 80.0,
    'english_score': 85.0
}

print("="*60)
print("STUDENT PERFORMANCE PREDICTION")
print("="*60)

print("\n📝 Student Information:")
for key, value in example_student.items():
    print(f"   {key}: {value}")

grade, probabilities = predict_student_performance(example_student)

print(f"\n🎯 Predicted Grade: {grade.upper()}")
print(f"\n📊 Confidence Scores:")
for i, prob in enumerate(probabilities):
    grade_name = ['F', 'E', 'D', 'C', 'B', 'A'][i]
    print(f"   {grade_name}: {prob:.2%}")
