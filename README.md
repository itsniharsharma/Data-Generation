# Data Generation Using Modelling and Simulation for Machine Learning

This project demonstrates an **end-to-end modelling, simulation, and machine learning pipeline** for generating synthetic data using a valid simulation tool and performing **model selection using TOPSIS**.

The objective is to show how **simulation-generated data** can be effectively used for **machine learning model evaluation and decision-making**, strictly following the assignment guidelines.

---

## Simulation Tool Used

**Simulator:** **SimPy**

SimPy is an open-source, process-based **discrete-event simulation framework for Python**.  
It is officially listed on Wikipedia under:

**List of Computer Simulation Software**  
https://en.wikipedia.org/wiki/List_of_computer_simulation_software

This satisfies the mandatory requirement that the simulator must be selected only from the provided Wikipedia list.

---

## Problem Description

A **queueing system** is simulated to model a real-world service process such as:
- Customer service desks
- Bank counters
- Call centers

The simulation captures how different system parameters affect **average customer waiting time**, which is later predicted using machine learning models.

---

## Simulation Model

### System Behavior
- Customers arrive randomly into the system
- A fixed number of servers provide service
- If all servers are busy, customers wait in a queue
- The simulation records the **average waiting time**

---

## Simulation Parameters and Bounds

The following parameters were randomized for each simulation run:

| Parameter | Description | Lower Bound | Upper Bound |
|--------|------------|------------|------------|
| arrival_rate | Mean customer arrival interval | 1 | 5 |
| service_rate | Mean service time | 1 | 6 |
| servers | Number of service counters | 1 | 5 |
| max_customers | Customers simulated | 50 | 200 |

These bounds ensure realistic and stable simulation behavior.

---

## Data Generation Methodology

- Random values were sampled uniformly within the defined parameter bounds
- Each parameter set was passed to the SimPy simulation
- The simulation returned the **average waiting time**
- This process was repeated **1000 times**

Each simulation run produced one data point.

### Generated Dataset
The dataset contains:
- Simulation input parameters
- Corresponding average waiting time

Saved at:

data/simpy_dataset.csv


---

## Machine Learning Problem Formulation

**Problem Type:** Regression  

**Objective:**  
Predict the average waiting time based on simulation parameters.

### Features
- arrival_rate
- service_rate
- servers
- max_customers

### Target Variable
- avg_wait_time

---

## Machine Learning Models Evaluated

A total of **8 regression models** were trained and evaluated:

1. Linear Regression  
2. Ridge Regression  
3. Lasso Regression  
4. K-Nearest Neighbors (KNN) Regressor  
5. Support Vector Regressor (SVR)  
6. Decision Tree Regressor  
7. Random Forest Regressor  
8. Gradient Boosting Regressor  

---

## Evaluation Metrics

Each model was evaluated using the following metrics:

| Metric | Type | Description |
|------|------|------------|
| MSE | Cost | Mean Squared Error |
| MAE | Cost | Mean Absolute Error |
| R² | Benefit | Coefficient of Determination |

- Lower values are better for **MSE** and **MAE**
- Higher values are better for **R²**

Evaluation results are saved in:

results/model_comparison.csv


---

## TOPSIS-Based Model Selection

To objectively rank models considering multiple evaluation metrics, **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)** was applied.

### TOPSIS Configuration
- Criteria:
  - MSE (Cost)
  - MAE (Cost)
  - R² (Benefit)
- Weights:
  - MSE: 0.33
  - MAE: 0.33
  - R²: 0.34

TOPSIS computes:
- Distance from ideal best solution
- Distance from ideal worst solution
- A final TOPSIS score for ranking models

### TOPSIS Results
Saved at:

results/topsis_ranking.csv


---

## Result Visualizations

The following plots were generated and saved automatically:

| Visualization | File |
|--------------|------|
| Arrival Rate vs Waiting Time | results/arrival_vs_wait.png |
| Model Comparison (R²) | results/model_r2_comparison.png |
| Model Comparison (MSE) | results/model_mse_comparison.png |
| TOPSIS Ranking | results/topsis_ranking.png |

These plots are used for analysis and reporting.

---

## Project Directory Structure

Data Generation using Modelling and Simulation for Machine Learning/
│
├── data/
│ └── simpy_dataset.csv
│
├── notebooks/
│ └── full_pipeline.ipynb
│
├── results/
│ ├── arrival_vs_wait.png
│ ├── model_comparison.csv
│ ├── model_mse_comparison.png
│ ├── model_r2_comparison.png
│ ├── topsis_ranking.csv
│ └── topsis_ranking.png
│
├── src/
│ ├── simulator.py
│ ├── data_generation.py
│ ├── ml_models.py
│ ├── topsis.py
│ └── utils.py
│
├── requirements.txt
├── README.md
└── venv/


---

## How to Run the Project

### Step 1: Activate Virtual Environment
```bash
source venv/bin/activate
```
Step 2: Run the Notebook

Open and execute:

notebooks/full_pipeline.ipynb
