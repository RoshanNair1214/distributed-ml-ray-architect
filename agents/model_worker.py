import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from ray import train

class ModelWorker:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_ray(self, config):
        """
        Distributed training function compatible with Ray Tune.
        Optimizes XGBoost hyperparameters.
        """
        # Initialize model with configuration from Ray Tune
        model = xgb.XGBRegressor(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 6),
            learning_rate=config.get("learning_rate", 0.1),
            subsample=config.get("subsample", 1.0),
            tree_method="auto" 
        )
        
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        
        # Calculate metrics for Ray to track
        rmse = np.sqrt(mean_squared_error(self.y_test, preds))
        r2 = r2_score(self.y_test, preds)
        
        # Report metrics back to the Ray scheduler
        train.report({"rmse": rmse, "r2": r2})

    def run_baseline(self):
        """Standard sequential training for performance benchmarking."""
        model = xgb.XGBRegressor()
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        return {
            "rmse": np.sqrt(mean_squared_error(self.y_test, preds)),
            "r2": r2_score(self.y_test, preds)
        }
