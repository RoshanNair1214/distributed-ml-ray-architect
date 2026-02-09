import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

class RayDistributor:
    def __init__(self):
        """Initializes the Ray runtime if not already active."""
        if not ray.is_initialized():
            # Standard initialization for local distributed processing
            ray.init(ignore_reinit_error=True)

    def orchestrate_hpo(self, train_func, num_samples=10):
        """
        Orchestrates the Distributed Hyperparameter Optimization (HPO).
        Uses the ASHA scheduler for early stopping of bad trials.
        """
        # Define the search space based on your AQMATiCS internship logic
        search_space = {
            "n_estimators": tune.randint(50, 500),
            "max_depth": tune.randint(3, 10),
            "learning_rate": tune.loguniform(1e-3, 0.1),
            "subsample": tune.uniform(0.5, 1.0)
        }

        # Setup the distributed tuner
        tuner = tune.Tuner(
            train_func,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                metric="rmse",
                mode="min",
                num_samples=num_samples,
                scheduler=ASHAScheduler() # Efficient resource allocation
            )
        )

        results = tuner.fit()
        return results.get_best_result(metric="rmse", mode="min")

    def shutdown(self):
        """Cleanly shuts down the Ray cluster resources."""
        ray.shutdown()
