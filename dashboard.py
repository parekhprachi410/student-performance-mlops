import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

API_URL = "http://localhost:8000"

def get_model_info():
    """Get model information"""
    response = requests.get(f"{API_URL}/model_info")
    if response.status_code == 200:
        return response.json()
    return None

def analyze_student_grades(students_data):
    """Analyze grade predictions for multiple students"""
    response = requests.post(f"{API_URL}/predict_batch", json={"students": students_data})
    if response.status_code == 200:
        return response.json()
    return None

# Example: Analyze a class of 10 students
class_students = [
    {
        "age": 16,
        "gender": "female",
        "school_type": "private",
        "parent_education": "graduate",
        "study_hours": 5.5,
        "attendance_percentage": 85.0,
        "internet_access": "yes",
        "travel_time": "15-30 min",
        "extra_activities": "yes",
        "study_method": "notes",
        "math_score": 75.0,
        "science_score": 80.0,
        "english_score": 85.0
    },
    {
        "age": 15,
        "gender": "male",
        "school_type": "public",
        "parent_education": "high school",
        "study_hours": 2.0,
        "attendance_percentage": 60.0,
        "internet_access": "no",
        "travel_time": ">60 min",
        "extra_activities": "no",
        "study_method": "textbook",
        "math_score": 45.0,
        "science_score": 50.0,
        "english_score": 55.0
    },
    {
        "age": 17,
        "gender": "other",
        "school_type": "private",
        "parent_education": "phd",
        "study_hours": 7.5,
        "attendance_percentage": 95.0,
        "internet_access": "yes",
        "travel_time": "<15 min",
        "extra_activities": "yes",
        "study_method": "online videos",
        "math_score": 92.0,
        "science_score": 88.0,
        "english_score": 95.0
    },
    {
        "age": 14,
        "gender": "male",
        "school_type": "public",
        "parent_education": "diploma",
        "study_hours": 1.5,
        "attendance_percentage": 45.0,
        "internet_access": "no",
        "travel_time": "30-60 min",
        "extra_activities": "no",
        "study_method": "group study",
        "math_score": 35.0,
        "science_score": 40.0,
        "english_score": 38.0
    },
    {
        "age": 18,
        "gender": "female",
        "school_type": "public",
        "parent_education": "post graduate",
        "study_hours": 6.0,
        "attendance_percentage": 88.0,
        "internet_access": "yes",
        "travel_time": "15-30 min",
        "extra_activities": "yes",
        "study_method": "coaching",
        "math_score": 82.0,
        "science_score": 85.0,
        "english_score": 88.0
    }
]

print("="*60)
print("STUDENT PERFORMANCE DASHBOARD")
print("="*60)

# Get model info
print("\n🤖 Model Information:")
model_info = get_model_info()
if model_info:
    for key, value in model_info.items():
        print(f"   {key}: {value}")

# Analyze class
print(f"\n📊 Analyzing {len(class_students)} students...")
results = analyze_student_grades(class_students)

if results:
    print(f"\n✅ Results:")
    print(f"   Total Processed: {results['total_processed']}")
    print(f"\n   Grade Distribution:")
    for grade, count in results['summary'].items():
        if count > 0:
            print(f"      {grade.upper()}: {count} student(s)")
    
    # Create visualization
    grades = list(results['summary'].keys())
    counts = list(results['summary'].values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(grades, counts, color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6', '#95a5a6'])
    plt.title('Predicted Grade Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Grade', fontsize=12)
    plt.ylabel('Number of Students', fontsize=12)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('grade_distribution_dashboard.png', dpi=100)
    print(f"\n   📊 Visualization saved to: grade_distribution_dashboard.png")
    plt.show()
    
    # Display individual predictions
    print(f"\n📝 Individual Student Predictions:")
    for pred in results['predictions']:
        print(f"\n   Student {pred['student_id']}:")
        print(f"      Predicted Grade: {pred['predicted_grade']}")
        print(f"      Risk Level: {pred['risk_level']}")
        print(f"      Recommendation: {pred['recommendation']}")
        print(f"      Confidence: {max(pred['confidence_scores'].values()):.1%}")

print("\n" + "="*60)
