[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Sysvkt7d8UAY9bkLVwXdw81gyJloBRKw#scrollTo=1jNvlQ_QzAD6)

Architecting Scalable Hyperparameter Optimization for High-Performance Workloads.

# Distributed-ml-RAY-architect
Distributed machine learning pipeline using Ray for high-performance hyperparameter tuning (HPO) and task orchestration, optimized for scalable XGBoost workflows.
# Distributed ML Architecture with Ray
---

## Project Overview
I identified a significant bottleneck in the model development lifecycle: sequential hyperparameter tuning. Traditional grid-search methods scale poorly as the complexity of the feature space increases. 

This project demonstrates a transition from localized, single-threaded model training to a **Distributed Computing Architecture**. By leveraging **Ray**, I parallelized the execution of independent numerical tasks and hyperparameter sweeps, reducing the computational time required to find the "Global Minimum" for model error.

---

## Methodology & Agent Logic

The system is modularized into specialized logic layers (Agents) to ensure maintainability and scalability:

### 1. Model Worker Agent (`model_worker.py`)
* **Task:** Encapsulates the core ML logic using **XGBoost**.
* **Role:** Acts as a remote worker that executes individual training trials. It is designed to be "Ray-aware," reporting metrics (RMSE/R²) back to the head node in real-time.

### 2. Ray Distributor Agent (`distributor.py`)
* **Task:** Orchestrates the distributed lifecycle.
* **Role:** Manages the **Ray Runtime** and defines the search space for Hyperparameter Optimization (HPO). It implements the **ASHA Scheduler**, an asynchronous successing halving algorithm that terminates underperforming trials early to save compute resources.

### 3. Performance Analyst Agent
* **Task:** Benchmarking and Validation.
* **Role:** Compares the **Baseline (Sequential)** performance against the **Ray-Optimized (Distributed)** results to verify the 2.1x speedup and the 8.5% reduction in error.

---

## Tech Stack
* **Distributed Computing:** [Ray](https://www.ray.io/) (Tasks, Actors, Tune)
* **Machine Learning:** XGBoost, Scikit-Learn
* **Frameworks:** Python, Pandas, NumPy
* **Optimization Logic:** Asynchronous Successive Halving Algorithm (ASHA)

---

## Key Results
* **Total Error Reduction:** -8.5% RMSE (from 4804.75 to 4394.81).
* **Speed Efficiency:** Reduced training time from 0.17s to 0.08s per trial.
* **Scalability:** Successfully processed a complex search space across multiple virtualized CPU cores.

---

### How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the main pipeline: `python main.py`
3. View detailed metrics in [RESULTS.md](./RESULTS.md)

