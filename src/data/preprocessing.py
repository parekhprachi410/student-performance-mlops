import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import yaml

class StudentDataPreprocessor:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_cols = self.config['features']['categorical']
        self.numerical_cols = self.config['features']['numerical']
        
    def load_and_explore(self, filepath):
        """Load dataset and perform initial exploration"""
        df = pd.read_csv(filepath)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Missing values:\n{df.isnull().sum()}")
        print(f"Final grade distribution:\n{df['final_grade'].value_counts()}")
        
        return df
    
    def clean_data(self, df):
        """Handle missing values and outliers"""
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            df = df.dropna()
            print(f"Removed {df.isnull().sum().sum()} missing values")
        
        # Remove outliers using IQR method for numerical features
        for col in self.numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            initial_len = len(df)
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            if len(df) < initial_len:
                print(f"Removed {initial_len - len(df)} outliers from {col}")
        
        return df
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        for col in self.categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            self.label_encoders[col] = le
            
        # Encode target variable (final_grade)
        target_encoder = LabelEncoder()
        df_encoded['final_grade_encoded'] = target_encoder.fit_transform(df_encoded['final_grade'])
        self.label_encoders['final_grade'] = target_encoder
        
        return df_encoded
    
    def create_features(self, df):
        """Create new features"""
        df['average_score'] = (df['math_score'] + df['science_score'] + df['english_score']) / 3
        df['total_score'] = df['math_score'] + df['science_score'] + df['english_score']
        df['study_efficiency'] = df['average_score'] / (df['study_hours'] + 1)
        df['attendance_score'] = df['attendance_percentage'] / 100
        
        # Grade mapping for regression
        grade_map = {'f': 0, 'e': 1, 'd': 2, 'c': 3, 'b': 4, 'a': 5}
        df['final_grade_numeric'] = df['final_grade'].map(grade_map)
        
        return df
    
    def prepare_features(self, df, target='final_grade_encoded'):
        """Prepare features for model training"""
        feature_cols = self.categorical_cols + self.numerical_cols + [
            'average_score', 'total_score', 'study_efficiency', 'attendance_score'
        ]
        
        X = df[feature_cols]
        y = df[target]
        
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
        
        print(f"Train size: {len(X_train)}")
        print(f"Validation size: {len(X_val)}")
        print(f"Test size: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessor(self, path='models/artifacts/preprocessor.pkl'):
        """Save preprocessor artifacts"""
        artifacts = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'categorical_cols': self.categorical_cols,
            'numerical_cols': self.numerical_cols
        }
        joblib.dump(artifacts, path)
        print(f"Preprocessor saved to {path}")

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = StudentDataPreprocessor()
    df = preprocessor.load_and_explore('data/raw/Student_Performance.csv')
    df = preprocessor.clean_data(df)
    df = preprocessor.create_features(df)
    df_encoded = preprocessor.encode_categorical(df)
    X, y, feature_cols = preprocessor.prepare_features(df_encoded)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    preprocessor.save_preprocessor()
    print("✅ Data preprocessing completed successfully!")
