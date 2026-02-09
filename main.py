from agents.model_worker import ModelWorker
from agents.distributor import RayDistributor

def main():
    # 1. Initialize Worker with your data
    worker = ModelWorker(X_train, y_train, X_test, y_test)
    
    # 2. Start Ray Distributor
    distributor = RayDistributor()
    
    # 3. Run Distributed HPO
    best_result = distributor.orchestrate_hpo(worker.train_ray, num_samples=20)
    
    print(f"Best Config Found: {best_result.config}")
    print(f"Best RMSE: {best_result.metrics['rmse']:.4f}")
    
    distributor.shutdown()

if __name__ == "__main__":
    main()
