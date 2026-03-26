
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import optuna
import numpy as np
import joblib
import yaml

class ModelTrainer:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
    def train_random_forest(self, X_train, y_train, X_val, y_val, params=None):
        """Train Random Forest model"""
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'random_state': self.config['model']['random_seed']
            }
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        print(f"Random Forest - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        
        return model, accuracy, f1
    
    def train_xgboost(self, X_train, y_train, X_val, y_val, params=None):
        """Train XGBoost model"""
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': self.config['model']['random_seed']
            }
        
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        print(f"XGBoost - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        
        return model, accuracy, f1
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, model_type='rf', n_trials=10):
        """Hyperparameter tuning using Optuna"""
        
        def objective(trial):
            if model_type == 'rf':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'random_state': self.config['model']['random_seed']
                }
                model = RandomForestClassifier(**params)
            else:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'random_state': self.config['model']['random_seed']
                }
                model = XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss')
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return f1_score(y_val, y_pred, average='weighted')
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"Best F1 score: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
        
        return study.best_params, study.best_value
    
    def train_with_mlflow(self, X_train, y_train, X_val, y_val, model_type='rf'):
        """Train model with MLflow tracking"""
        
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))
            
            # Hyperparameter tuning (reduced trials for speed)
            best_params, best_f1 = self.hyperparameter_tuning(
                X_train, y_train, X_val, y_val, model_type, n_trials=10
            )
            
            # Log best parameters
            for param, value in best_params.items():
                mlflow.log_param(f"best_{param}", value)
            
            # Train final model
            if model_type == 'rf':
                model = RandomForestClassifier(**best_params, random_state=self.config['model']['random_seed'])
            else:
                model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='mlogloss')
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            
            # Log classification report
            report = classification_report(y_val, y_pred, output_dict=True)
            mlflow.log_dict(report, "classification_report.json")
            
            # Log model
            mlflow.sklearn.log_model(model, f"{model_type}_model")
            
            print(f"✅ {model_type.upper()} model trained - F1 Score: {f1:.4f}")
            
            return model, run.info.run_id
    
    def ensemble_predict(self, models, X):
        """Ensemble prediction from multiple models"""
        predictions = []
        for model in models:
            predictions.append(model.predict_proba(X))
        
        # Average probabilities
        avg_probs = np.mean(predictions, axis=0)
        return np.argmax(avg_probs, axis=1)

if __name__ == "__main__":
    print("Model trainer module loaded successfully!")
