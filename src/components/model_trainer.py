import os
import sys
import pickle
import yaml
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils


@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join("artifacts")
    trained_model_path = os.path.join(artifact_folder, "model.pkl")
    expected_accuracy = 0.45
    model_config_file_path = os.path.join('config', 'model.yaml')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()

        # Define models with initial parameters
        self.models = {
            'XGBClassifier': XGBClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'SVC': SVC(),
            'RandomForestClassifier': RandomForestClassifier()
        }

        # Default hyperparameters for tuning
        self.param_grid = {
            'XGBClassifier': {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]},
            'GradientBoostingClassifier': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
            'SVC': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
            'RandomForestClassifier': {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
        }

    def load_model_config(self) -> dict:
        """
        Load the model configuration from a YAML file.
        """
        try:
            with open(self.model_trainer_config.model_config_file_path, 'r') as file:
                model_config = yaml.safe_load(file)
            return model_config
        except Exception as e:
            raise CustomException(e, sys)

    def fine_tune_model(self, model, param_grid, X_train, y_train):
        """
        Fine-tune the model using GridSearchCV.
        """
        try:
            logging.info(f"Fine-tuning model: {model.__class__.__name__}")
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=2)
            grid_search.fit(X_train, y_train)
            logging.info(f"Best parameters for {model.__class__.__name__}: {grid_search.best_params_}")
            return grid_search.best_estimator_
        except Exception as e:
            raise CustomException(e, sys)

    def train_and_evaluate(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Tuple[str, float, object]:
        """
        Train and evaluate models, fine-tune them, and return the best model.
        """
        try:
            best_model_name = None
            best_model = None
            best_accuracy = 0

            for model_name, model in self.models.items():
                logging.info(f"Training model: {model_name}")

                # Perform fine-tuning if hyperparameters are available
                if model_name in self.param_grid:
                    model = self.fine_tune_model(model, self.param_grid[model_name], X_train, y_train)

                # Train and evaluate
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                logging.info(f"{model_name} Accuracy: {accuracy}")

                if accuracy > best_accuracy and accuracy >= self.model_trainer_config.expected_accuracy:
                    best_model_name = model_name
                    best_model = model
                    best_accuracy = accuracy

            if best_model is None:
                raise CustomException("No model achieved the expected accuracy.", sys)

            logging.info(f"Best Model: {best_model_name} with Accuracy: {best_accuracy}")
            return best_model_name, best_accuracy, best_model

        except Exception as e:
            raise CustomException(e, sys)

    def save_model(self, model: object):
        """
        Save the model to a pickle file.
        """
        try:
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)
            with open(self.model_trainer_config.trained_model_path, 'wb') as file:
                pickle.dump(model, file)
            logging.info(f"Model saved at {self.model_trainer_config.trained_model_path}")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        """
        Main function to initiate model training and save the best model.
        """
        try:
            logging.info("Extracting train and test arrays.")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1]
            )

            logging.info("Training and evaluating models.")
            _, _, best_model = self.train_and_evaluate(X_train, X_test, y_train, y_test)

            logging.info("Saving the best model.")
            self.save_model(best_model)

        except Exception as e:
            raise CustomException(e, sys)
