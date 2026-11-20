# Python Portfolio – Machine Learning & Engineering Modelling

Projects completed during my studies, implemented in Python using Jupyter notebooks.
Each folder contains the notebook and the data it uses.

**Tools:** Python, pandas, NumPy, scikit-learn, PyTorch, Matplotlib, seaborn

---

## 1. Bike-Share Demand Analysis
📂 [`bike_share`](bike_share/bike_share_prediction.ipynb)

Analysis of the Capital Bike-Share dataset (2011–2012, hourly rentals).
- Data preprocessing, visualisation and a chronological train/validation/test split (no data leakage)
- Linear regression to predict hourly rentals; adding interaction terms (hour × temperature, etc.) improved R² from 0.298 to 0.384 and reduced RMSE from 184.8 to 173.0
- Residual diagnostics, separate models for casual vs. registered users
- Logistic regression for classifying high-demand periods, evaluated with accuracy, precision, recall and confusion matrices

**Key insight:** demand is bimodal, peaking at 8:00 and 17:00–18:00 commuter hours; warmer temperatures increase rentals, especially in the evening.

## 2. Satellite Image Classification
📂 [`satellite_classification`](satellite_classification/satellite_image_classification.ipynb)

Binary classification on a 36-feature satellite imagery dataset (spectral measurements of neighbouring pixels).
- Exploratory analysis: correlation heatmap, class distributions
- Compared Logistic Regression, Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA) using ROC curves and AUC
- QDA had the highest training AUC (0.965) but overfitted (lowest validation AUC, 0.892); simpler models generalised better
- Cost-sensitive threshold optimisation: when a false negative costs 5× a false positive, the optimal threshold drops to 0.09, giving 99.6% recall

## 3. Network Traffic Classification (Cyberattack Detection)
📂 [`traffic_prediction`](traffic_prediction/traffic_prediction.ipynb)

Detecting cyberattacks in network traffic data (~146,000 connections) – binary (normal vs. attack) and multiclass (normal, DoS, probe, R2L, U2R).
- Decision trees with cross-validated depth tuning and cost-complexity pruning; feature importance analysis
- Effect of feature scaling: no impact on decision trees, but nearly doubled linear SVM macro F1 (0.439 → 0.869) and cut training time
- Regularisation tuning for linear SVM and comparison with RBF-kernel SVM (linear SVM performed better: macro F1 0.896 vs. 0.847)
- Class weighting for rare attacks: raised R2L recall from 91.5% to 97.0% and lowered expected cost
- Shallow neural network classifier built in PyTorch

## 4. Partial Differential Equation Models – Leopard Pattern Simulation
📂 [`models_engineering`](models_engineering/pde_models_engineering.ipynb)

Simulation of animal skin pigment patterns using a two-morphogen Turing reaction–diffusion model (Liu, Liaw & Maini, 2006).
- Implemented finite-difference discretisation of the PDE system in NumPy
- Stage 1: juvenile leopard spots formed from random noise
- Stage 2: transformation of spots into adult leopard rosettes
- Stage 3: labyrinth patterns obtained by changing the nonlinear parameter r₃
- 3D-style artistic visualisation with Matplotlib lighting; results validated against real leopard photos
