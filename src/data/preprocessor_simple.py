import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import yaml
import os

class SimpleDataPreprocessor:
    def __init__(self, config_path='config/config.yaml'):
        # Load config if exists
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                'features': {
                    'categorical': ['gender', 'school_type', 'parent_education', 
                                   'internet_access', 'travel_time', 'extra_activities', 
                                   'study_method'],
                    'numerical': ['age', 'study_hours', 'attendance_percentage',
                                 'math_score', 'science_score', 'english_score']
                }
            }
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_cols = self.config['features']['categorical']
        self.numerical_cols = self.config['features']['numerical']
        
    def load_and_explore(self, filepath):
        """Load dataset and perform initial exploration"""
        df = pd.read_csv(filepath)
        
        print(f"\n📊 Dataset Information:")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {len(df.columns)} features")
        print(f"\n📋 Column names:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. {col}")
        
        print(f"\n🔍 Missing Values:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("   No missing values found!")
        
        print(f"\n📈 Final Grade Distribution:")
        grade_counts = df['final_grade'].value_counts()
        for grade, count in grade_counts.items():
            print(f"   {grade.upper()}: {count:5d} students ({count/len(df)*100:.1f}%)")
        
        return df
    
    def clean_data(self, df):
        """Handle missing values and outliers"""
        # Remove rows with missing values
        initial_len = len(df)
        df = df.dropna()
        if len(df) < initial_len:
            print(f"   Removed {initial_len - len(df)} rows with missing values")
        
        # Remove outliers for numerical columns
        for col in self.numerical_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                initial_count = len(df)
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                if len(df) < initial_count:
                    print(f"   Removed {initial_count - len(df)} outliers from {col}")
        
        return df
    
    def create_features(self, df):
        """Create new features"""
        # Calculate average score
        df['average_score'] = (df['math_score'] + df['science_score'] + df['english_score']) / 3
        
        # Calculate total score
        df['total_score'] = df['math_score'] + df['science_score'] + df['english_score']
        
        # Calculate study efficiency
        df['study_efficiency'] = df['average_score'] / (df['study_hours'] + 1)
        
        # Calculate attendance score
        df['attendance_score'] = df['attendance_percentage'] / 100
        
        print(f"\n✨ Created new features:")
        print(f"   - average_score")
        print(f"   - total_score")
        print(f"   - study_efficiency")
        print(f"   - attendance_score")
        
        return df
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        for col in self.categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                print(f"   Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Encode target variable
        target_encoder = LabelEncoder()
        df_encoded['final_grade_encoded'] = target_encoder.fit_transform(df_encoded['final_grade'])
        self.label_encoders['final_grade'] = target_encoder
        
        print(f"\n🎯 Target encoded:")
        for i, grade in enumerate(target_encoder.classes_):
            print(f"   {grade.upper()} -> {i}")
        
        return df_encoded
    
    def prepare_features(self, df, target='final_grade_encoded'):
        """Prepare features for model training"""
        feature_cols = (self.categorical_cols + self.numerical_cols + 
                       ['average_score', 'total_score', 'study_efficiency', 'attendance_score'])
        
        # Only use columns that exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols]
        y = df[target]
        
        print(f"\n🔧 Feature preparation:")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Samples: {len(X)}")
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, feature_cols
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessor(self, path='models/artifacts/preprocessor.pkl'):
        """Save preprocessor artifacts"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        artifacts = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'categorical_cols': self.categorical_cols,
            'numerical_cols': self.numerical_cols
        }
        joblib.dump(artifacts, path)
        print(f"\n💾 Preprocessor saved to: {path}")

if __name__ == "__main__":
    print("="*60)
    print("TESTING DATA PREPROCESSOR")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = SimpleDataPreprocessor()
    
    # Load data
    df = preprocessor.load_and_explore('data/raw/Student_Performance.csv')
    
    # Clean data
    print("\n🧹 Cleaning data...")
    df = preprocessor.clean_data(df)
    
    # Create features
    print("\n🔨 Creating features...")
    df = preprocessor.create_features(df)
    
    # Encode categorical
    print("\n🏷️ Encoding categorical variables...")
    df_encoded = preprocessor.encode_categorical(df)
    
    # Prepare features
    X, y, feature_cols = preprocessor.prepare_features(df_encoded)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    print("\n" + "="*60)
    print("✅ DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)
