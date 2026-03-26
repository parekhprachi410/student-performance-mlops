import requests
import json
import sys

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        print(f"✅ Health Check: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure it's running on port 8000")
        return False

def test_predict():
    """Test single prediction"""
    student_data = {
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
    }
    
    print("\n📝 Testing Single Prediction...")
    
    try:
        response = requests.post(f"{API_URL}/predict", json=student_data, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Prediction Successful!")
            print(f"🎯 Predicted Grade: {result['predicted_grade']}")
            print(f"⚠️ Risk Level: {result['risk_level']}")
            print(f"💡 Recommendation: {result['recommendation']}")
            print(f"\n📈 Confidence Scores:")
            for grade, score in result['confidence_scores'].items():
                print(f"   {grade}: {score:.2%}")
            return True
        else:
            print(f"\n❌ Prediction Failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_batch_predict():
    """Test batch prediction"""
    batch_data = {
        "students": [
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
            }
        ]
    }
    
    print("\n📊 Testing Batch Prediction...")
    print(f"Number of students: {len(batch_data['students'])}")
    
    try:
        response = requests.post(f"{API_URL}/predict_batch", json=batch_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Batch Prediction Successful!")
            print(f"Total Processed: {result['total_processed']}")
            print(f"\n📊 Risk Distribution:")
            for grade, count in result['summary'].items():
                if count > 0:
                    print(f"   {grade.upper()}: {count} student(s)")
            
            print(f"\n🎯 First Student Details:")
            first = result['predictions'][0]
            print(f"   Grade: {first['predicted_grade']}")
            print(f"   Risk: {first['risk_level']}")
            return True
        else:
            print(f"\n❌ Batch Prediction Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("API TEST SUITE")
    print("="*60)
    
    if not test_health():
        print("\n❌ API is not running. Start it with: uvicorn src.api.app:app --reload")
        sys.exit(1)
    
    test_predict()
    test_batch_predict()
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
