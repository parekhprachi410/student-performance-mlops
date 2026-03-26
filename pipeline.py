import mlflow
from src.data.preprocessing import StudentDataPreprocessor
from src.models.train import ModelTrainer
import joblib
import yaml

def run_training_pipeline(config_path='config/config.yaml'):
    """Complete MLOps training pipeline"""
    
    print("="*60)
    print("STUDENT PERFORMANCE PREDICTION - TRAINING PIPELINE")
    print("="*60)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Step 1: Initialize components
    print("\n📊 Step 1: Initializing preprocessor...")
    preprocessor = StudentDataPreprocessor(config_path)
    trainer = ModelTrainer(config_path)
    
    # Step 2: Load and preprocess data
    print("\n📂 Step 2: Loading and preprocessing data...")
    df = preprocessor.load_and_explore(config['data']['raw_path'])
    df = preprocessor.clean_data(df)
    df = preprocessor.create_features(df)
    df_encoded = preprocessor.encode_categorical(df)
    
    # Step 3: Prepare features
    print("\n🔧 Step 3: Preparing features...")
    X, y, feature_cols = preprocessor.prepare_features(df_encoded)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X, y, test_size=config['model']['test_size'], 
        val_size=config['model']['val_size']
    )
    
    # Step 4: Train models
    print("\n🤖 Step 4: Training models...")
    models = {}
    
    # Random Forest
    print("\n🌲 Training Random Forest...")
    rf_model, rf_run_id = trainer.train_with_mlflow(
        X_train, y_train, X_val, y_val, model_type='rf'
    )
    models['random_forest'] = rf_model
    
    # XGBoost
    print("\n🚀 Training XGBoost...")
    xgb_model, xgb_run_id = trainer.train_with_mlflow(
        X_train, y_train, X_val, y_val, model_type='xgb'
    )
    models['xgboost'] = xgb_model
    
    # Step 5: Ensemble predictions
    print("\n🎯 Step 5: Creating ensemble predictions...")
    y_pred_ensemble = trainer.ensemble_predict(
        [rf_model, xgb_model], X_test
    )
    
    # Step 6: Evaluate ensemble
    print("\n📈 Step 6: Evaluating ensemble model...")
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    accuracy = accuracy_score(y_test, y_pred_ensemble)
    f1 = f1_score(y_test, y_pred_ensemble, average='weighted')
    
    print(f"\n✅ Ensemble Model Performance:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    
    # Step 7: Save best model
    print("\n💾 Step 7: Saving best model...")
    best_model = rf_model if f1 > 0.85 else xgb_model
    joblib.dump(best_model, 'models/artifacts/best_model.pkl')
    preprocessor.save_preprocessor()
    
    # Log final metrics
    with mlflow.start_run(run_name="final_model"):
        mlflow.log_metric("ensemble_accuracy", accuracy)
        mlflow.log_metric("ensemble_f1", f1)
        mlflow.sklearn.log_model(best_model, "best_model")
    
    print("\n✅ Training pipeline completed successfully!")
    print(f"📁 Best model saved to: models/artifacts/best_model.pkl")
    print(f"📁 Preprocessor saved to: models/artifacts/preprocessor.pkl")
    print(f"📊 MLflow runs logged in: mlruns/")
    
    return best_model, preprocessor

if __name__ == "__main__":
    model, preprocessor = run_training_pipeline()
