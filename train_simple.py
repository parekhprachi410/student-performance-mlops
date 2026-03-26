import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("STUDENT PERFORMANCE PREDICTION - SIMPLE TRAINING")
print("="*60)

# Import our preprocessor
from src.data.preprocessor_simple import SimpleDataPreprocessor

# Step 1: Preprocess data
print("\n📊 Step 1: Preprocessing data...")
preprocessor = SimpleDataPreprocessor()
df = preprocessor.load_and_explore('data/raw/Student_Performance.csv')
df = preprocessor.clean_data(df)
df = preprocessor.create_features(df)
df_encoded = preprocessor.encode_categorical(df)
X, y, feature_cols = preprocessor.prepare_features(df_encoded)
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)

# Step 2: Train Random Forest
print("\n🌲 Step 2: Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Step 3: Evaluate
print("\n📈 Step 3: Evaluating model...")
y_pred = rf_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"\n✅ Validation Results:")
print(f"   Accuracy: {accuracy:.4f}")
print(f"\n📊 Classification Report:")
print(classification_report(y_val, y_pred, target_names=preprocessor.label_encoders['final_grade'].classes_))

# Step 4: Test on test set
print("\n🎯 Step 4: Testing on test set...")
y_test_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"   Test Accuracy: {test_accuracy:.4f}")

# Step 5: Feature importance
print("\n🔍 Step 5: Feature Importance Analysis...")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Top 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:25s}: {row['importance']:.4f}")

# Step 6: Save model
print("\n💾 Step 6: Saving model...")
joblib.dump(rf_model, 'models/artifacts/simple_model.pkl')
preprocessor.save_preprocessor()

print("\n" + "="*60)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"📁 Model saved to: models/artifacts/simple_model.pkl")
print(f"📁 Preprocessor saved to: models/artifacts/preprocessor.pkl")
