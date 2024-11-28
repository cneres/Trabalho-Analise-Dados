from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import pandas as pd
from datetime import datetime

class MLModelHandler:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier,
            'logistic_regression': LogisticRegression
        }
        self.current_model = None
        self.current_model_type = None
        self.model_path = 'models/trained/'
        os.makedirs(self.model_path, exist_ok=True)

    def train_model(self, data, model_type='random_forest', parameters=None):
       
        try:
            df = pd.DataFrame(data)
            target_column = 'target'  

            if target_column not in df.columns:
                return {"error": "Target column not found in data."}

            X = df.drop(columns=[target_column])
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if parameters is None:
                parameters = {}

            model_class = self.models.get(model_type)
            if model_class is None:
                return {"error": f"Model type {model_type} not supported"}

            self.current_model = model_class(**parameters)
            self.current_model_type = model_type

            self.current_model.fit(X_train, y_train)

            predictions = self.current_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)

            self.save_model()

            return {
                "accuracy": accuracy,
                "report": report,
                "predictions": predictions.tolist()
            }

        except Exception as e:
            return {"error": str(e)}

    def predict(self, data):
     
        if self.current_model is None:
            return {"error": "No model currently loaded"}

        try:
            df = pd.DataFrame(data)
            predictions = self.current_model.predict(df)
            return {"predictions": predictions.tolist()}
        except Exception as e:
            return {"error": str(e)}

    def save_model(self):
        """Save current model to disk"""
        if self.current_model is None:
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.current_model_type}_{timestamp}.joblib"
        filepath = os.path.join(self.model_path, filename)
        joblib.dump(self.current_model, filepath)
        return True

    def load_latest_model(self, model_type=None):
        
        if model_type is None and self.current_model_type is None:
            return False

        model_type = model_type or self.current_model_type
        model_files = [f for f in os.listdir(self.model_path) 
                      if f.startswith(model_type)]

        if not model_files:
            return False

        latest_model = max(model_files)
        self.current_model = joblib.load(
            os.path.join(self.model_path, latest_model)
        )
        self.current_model_type = model_type
        return True

    def clear_model_cache(self):
     
        for file in os.listdir(self.model_path):
            os.remove(os.path.join(self.model_path, file))
        self.current_model = None
        self.current_model_type = None
        return True