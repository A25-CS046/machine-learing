﻿![Python](https://img.shields.io/badge/Python-3670A0?s&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?s&logo=TensorFlow&logoColor=white)
![Maintenance](https://img.shields.io/badge/Maintenance-Yes-green)
![Build](https://img.shields.io/badge/Build-Passing-green)

<div align="center">
  <img width="1584" height="396" alt="aegis-banner" src="https://github.com/user-attachments/assets/05cafb17-789c-4738-b699-96387a28b800" />
  <h3>AEGIS (AI Engine for Grounded Inspection System)</h3>
  <h4>A25-CS046 Capstone Project - Predictive Maintenance Copilot</h4>
</div>

### Project Overview
<p align="justify">
This repository implements a comparative analysis between LSTM and XGBoost models for anomaly detection and time-series prediction in industrial machine data. Both models are evaluated for detecting potential equipment failures and forecasting sensor behavior over time.
</p>

<p align="justify">
The pipeline aims to identify early signs of machine failure from sensor data (temperature, torque, rpm, etc.) and forecast future sensor conditions to enable proactive maintenance scheduling.
</p>

Both models are tested under identical preprocessing and evaluation conditions to assess:
- Predictive performance (classification metrics)
- Temporal sensitivity (sequence handling)
- Suitability for different dataset structures (anti-leakage vs synthesized)

### Data Understanding
<p align="justify">
    The project utilizes two distinct datasets derived from the AI4I 2020 Predictive Maintenance Dataset. While these datasets simulate real-world industrial milling operations, it is crucial to note that both are entirely synthetic.
    They were generated using a simulation model to replicate the degradation and failure patterns of a CNC machine tool, providing a controlled environment for testing predictive maintenance algorithms.
</p>

<b>The Original Dataset</b> `predictive_maintenance.csv`
<p align="justify">
    This dataset represents a "static snapshot" approach. It contains 10,000 data points, where each row represents a single, independent state of a machine at a specific moment.
</p>

- Sources: [Click here!](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)
- Nature: Cross-sectional data (snapshots).
- Use Case: Binary Classification (Fail vs. No Fail) or Multi-class Classification (classifying the specific type of failure).
- Key Characteristic: There is no temporal continuity between rows; Row 1 is not necessarily followed by Row 2 in time.
  
**Feature Description**:
<div align=center>
    
| Column | Description |
|---------|--------------|
| **UDI** | Unique identifier ranging from 1 to 10,000. |
| **Product ID** | Encodes product quality (L, M, H = low, medium, high) and a variant serial number. |
| **Type** | Machine or product quality class - L (Low), M (Medium), or H (High). |
| **Air temperature [K]** | Simulated via random walk, mean 300 K, standard deviation ±2 K. |
| **Process temperature [K]** | Air temperature +10 K offset with an additional ±1 K random walk. |
| **Rotational speed [rpm]** | Derived from 2860 W nominal power with added Gaussian noise. |
| **Torque [Nm]** | Normally distributed around 40 Nm, σ = 10 Nm, non-negative. |
| **Tool wear [min]** | Cumulative wear depending on product quality: +5 (H), +3 (M), +2 (L). |
| **Target** | **Primary target** - binary failure indicator (`1 = failure`, `0 = normal operation`). Used for anomaly detection and failure prediction tasks. |
| **Failure Type** | **Secondary target** - categorical label describing *failure mode* when `Target = 1`. Classes include: <br> • No Failure <br> • Power Failure <br> • Tool Wear Failure <br> • Overstrain Failure <br> • Heat Dissipation Failure <br> • Random Failure |

</div>

<b>The Time-Series Dataset</b> `predictive_maintenance_timeseries.csv`
<p align="justify">
    This dataset expands upon the concepts of the original by introducing a temporal dimension. It tracks 400 unique units over time, recording their sensor readings at regular intervals (steps) until a failure occurs or the observation ends.
</p>

- Nature: Longitudinal/Time-series data.
- Use Case: Remaining Useful Life (RUL) estimation and sequence modeling (e.g., using LSTMs).
- Key Characteristic: Contains a `step_index` and `synthetic_RUL` (Remaining Useful Life), allowing models to learn the *trajectory* of degradation rather than just the final failure state.

**Feature Description**:
<div align=center>

| Feature Name | Type | Dataset | Description |
| :--- | :--- | :--- | :--- |
| **UDI / Unit ID** | Identifier | Both | A unique identifier for the specific machine or observation sequence. |
| **Product ID** | Categorical | Both | Identifies the product quality variant: **L** (Low, 50%), **M** (Medium, 30%), or **H** (High, 20%). |
| **Type** | Categorical | Original | The quality variant ('L', 'M', 'H') extracted as a distinct category. |
| **Air temperature [K]** | Numerical | Both | The ambient room temperature, normalized around 300K. |
| **Process temperature [K]** | Numerical | Both | The temperature of the manufacturing process itself; typically 10K higher than air temperature. |
| **Rotational speed [rpm]** | Numerical | Both | The speed at which the spindle is rotating (calculated from power & torque). |
| **Torque [Nm]** | Numerical | Both | The torque force applied by the spindle. Normally distributed around 40 Nm. |
| **Tool wear [min]** | Numerical | Both | The accumulated usage time of the cutting tool. Failures are more likely as this increases. |
| **Target / Is Failure** | Target (Class) | Both | **0** = No Failure, **1** = Failure. The primary target for classification tasks. |
| **Failure Type** | Target (Class) | Both | The specific reason for failure: *Heat Dissipation, Power Failure, Overstrain, Tool Wear,* or *No Failure*. |
| **Step Index** | Numerical | Time-series | The chronological time-step of the observation for a specific Unit ID. |
| **Synthetic RUL** | Target (Reg) | Time-series | **Remaining Useful Life**. The number of time steps remaining until the machine fails. |

</div>

#### Data Limitations & Constraints
Since the foundation of this project is synthetic data, several limitations must be acknowledged when interpreting model performance:

1.  **Idealized Physics & Clean Patterns**
    * The data is generated using simplified mathematical formulas (e.g., `Power = Torque * Speed`). In real-world machinery, these relationships are often non-linear and affected by external factors like vibration, material inconsistency, or sensor drift.
    * **Impact:** Models like XGBoost may achieve unrealistically high accuracy (e.g., 99.5% AUC) because the "rules" of failure are mathematically encoded in the data itself, making them easier to reverse-engineer than real organic failures.

2.  **Lack of Real-World Noise**
    * Real industrial sensors are noisy. They have missing values, spikes, and electromagnetic interference. This synthetic dataset is "clean," lacking the chaotic noise floor typical of IoT environments.
    * **Impact:** A model trained here might be brittle when applied to real sensor data, as it hasn't learned to filter out noise.

3.  **Simplified Failure Modes**
    * The dataset contains only 5 specific failure types (plus "Random" in the original). In reality, machines can fail in infinite ways.
    * **Note:** The Time-Series dataset does *not* contain the "Random Failures" class found in the original dataset, making the time-series prediction task slightly more deterministic.

4.  **Synthetic Seasonality**
    * The "Timestamp" column in the time-series dataset is simulated. It does not reflect true seasonality (e.g., failures happening more often on hot summer days vs. winter) unless explicitly programmed into the simulation parameters.

**Note**:
<p align="justify">
    While the dataset is synthetic, its structure and statistical dynamics were carefully engineered to approximate real industrial sensor data used in predictive maintenance research and production analytics.
</p>

### Pipeline Architecture Diagram
<img width="4575" height="2119" alt="Document Systems" src="https://github.com/user-attachments/assets/c78e08da-38db-4c88-ba73-ce2dced19d6e" />

### Experimental Results & Performance Evaluation

<b>Classification Metrics</b>
<div align=center>
    
| Metric | XGBoost | LSTM | Difference |
| :--- | :--- | :--- | :--- |
| **AUC Score** | **0.995** | 0.828 | +0.167 |
| **Accuracy** | **0.988** | 0.875 | +0.113 |
| **Precision** | **0.933** | 0.247 | +0.686 |
| **Recall** | **1.000** | 0.467 | +0.533 |
| **F1 Score** | **0.966** | 0.323 | +0.643 |

</div>

<b>Regression Metrics (RUL Prediction)</b>

<div align=center>

| Metric | XGBoost | LSTM | Difference |
| :--- | :--- | :--- | :--- |
| **RMSE** (Lower is better) | **11.7** | 26.9 | -15.2 |
| **MAE** (Lower is better) | **9.5** | 20.9 | -11.4 |
| **MSE** (Lower is better) | **137.3** | 723.2 | -585.9 |

</div>

We have selected XGBoost as the production model for our Predictive Maintenance system. Based on the evaluation dashboard, XGBoost significantly outperforms the LSTM implementation across both classification (failure detection) and regression (Remaining Useful Life/RUL) tasks.
The decision is driven by the following critical factors:

1. Critical Failure Detection (Recall) In predictive maintenance, the cost of missing a failure (False Negative) is extremely high.
XGBoost achieved a Recall of 1.000, meaning it successfully identified 100% of the failure events in the test set.
LSTM achieved a Recall of only 0.467, meaning it failed to identify more than half of the potential equipment failures.

2. Reduction of False Alarms (Precision)
XGBoost (Precision: 0.933) creates a trustworthy system where alerts are highly likely to be genuine issues.
LSTM (Precision: 0.247) generated a significant number of false positives (298 false alarms vs. 98 true positives), which would lead to alert fatigue and wasted maintenance resources.

3. RUL Prediction Capability
XGBoost demonstrates a clear linear correlation between Predicted RUL and True RUL (RMSE: 11.72), indicating it has learned the degradation patterns of the machinery.
LSTM failed to capture the temporal dependencies, resulting in a "flat-line" prediction (visible in the LSTM RUL scatter plot) where it essentially predicted the mean value for all instances regardless of the actual machine state.

4. Model Robustness The XGBoost model shows an Area Under the Curve (AUC) of 0.995, indicating near-perfect separability between healthy and failing states, whereas the LSTM struggles significantly with an AUC of 0.828.

### Technology Stack
<p align="justify">
    This project leverages a modern Python-based stack, combining traditional Machine Learning for predictive maintenance with Generative AI for the Copilot interface.
</p>

Machine Learning & Data Science
- XGBoost (v2.0.3): The primary model used for production failure prediction and RUL estimation due to its high accuracy (AUC 0.995).
- TensorFlow & Keras: Used for developing the LSTM baseline model for time-series sequence comparison.
- Imbalanced-learn: Implemented SMOTE and sampling techniques to handle the rarity of failure events in the dataset.
- Optuna: Utilized for automated hyperparameter tuning to maximize model performance.
- SHAP: Provides model explainability, allowing the Copilot to explain why a specific machine was flagged as "at risk."
- Pandas & NumPy: Core data manipulation and numerical analysis.

Generative AI & Orchestration (Copilot)
- Google Gemini 2.5 Pro (via google-generativeai): The LLM engine powering the chatbot, capable of interpreting technical data and answering user queries.
- LangChain: Framework for orchestrating the flow between the user, the LLM, and the database/model inference results.
- LangSmith: Used for tracing and debugging LLM application flows.

Backend & Infrastructure
- Flask (v3.0.0): Lightweight WSGI web application framework serving the REST API for model inference and the chatbot.
- Gunicorn: Production-grade WSGI server to handle concurrent requests.
- PostgreSQL & SQLAlchemy: Relational database for storing user logs, chat history, and machine metadata.
- Cloud SQL: Hosting PostgrelSQL instances
- Google Cloud Storage: Cloud object storage for managing model artifacts (.pkl, .h5, .joblib) and datasets.
- Cloud Run: Using Cloud run for model server deployment

Visualization & Dashboarding
- Plotly & Seaborn: Used to generate interactive charts for the RUL degradation curves and confusion matrices.
- Matplotlib: Static plotting for report generation.

Testing & Quality Assurance
- Pytest: Framework for unit testing API endpoints and ML pipeline logic.
- Pytest-cov: Ensures high code coverage across the application.

### Copilot Server Installation
- For copilot features, API and complete installation guide please refer to [Predictive Maintenance Copilot Server](https://github.com/A25-CS046/machine-learning/blob/main/pm-app/README.md) documentation
- For a complete list of libraries and exact versions used in this projects, please refer to `requirements.txt` in the root folder and in the `pm-app/`.

### Contributor
<div align=center>

| Group | Name  | Asah ID | Roles | University |
|---|---|---|---|---|
| Machine Learning | Naufal Rahfi Anugerah | M254D5Y1475 | Project Manager, Machine Learning Engineer | Universitas Mercu Buana |
| Machine Learning | Clavino Ourizqi Rachmadi | M254D5Y0400	| Machine Learning Engineer | Universitas Mercu Buana |

</div>

### License
<p align="justify">
    There is NO LICENSE available yet as this project is still being used for purposes that cannot be published as open source, therefore please read the disclaimer section.
</p>

### Disclaimer
- This project is currently part of the [Asah led by Dicoding in association with Accenture](https://www.dicoding.com/asah) program and is being developed by our team as part of the Capstone Project.
- The primary purpose of this project is to fulfill the requirements of the Asah program and to demonstrate the technical and collaborative skills we have acquired during the program.
- The project is not yet intended for open-source release or commercial use. All rights and restrictions related to the project remain under the team's discretion until further notice.

### Author
Github Organization: [AEGIS - A25-CS046 Capstone Team](https://github.com/A25-CS046)
