#K-Nearest Neighbors (KNN) Classification – Iris Dataset
  This project demonstrates KNN Classification using the Iris dataset inside GitHub Codespaces.
  It includes feature scaling, training the model, tuning K, model evaluation, and visualizing decision boundaries.

Getting Started (Codespaces)
1. Open in Codespaces
    -Click Code → Create Codespace on main.

2. Install dependencies
    -Run in terminal:
   ##pip install -r requirements.txt

3. Launch Jupyter Notebook (optional)
    -jupyter notebook --ip 0.0.0.0 --no-browser

     *Tools and Libraries
      --Python
      --Pandas
      --NumPy
      --Matplotlib
      --Scikit-learn
      --GitHub Codespaces environment

"""Import Libraries
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.model_selection import train_test_split, cross_val_score
  from sklearn.preprocessing import StandardScaler
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.metrics import accuracy_score, confusion_matrix, classification_report """

*Dataset
  Dataset used: Iris Flower Dataset
  Source: https://www.kaggle.com/datasets/uciml/iris

  --Upload the file iris.csv to the Codespace root folder before running the scripts.

**Project Workflow
1. Load and Explore Data
  --View first rows, shape, descriptive statistics.

2. Normalize Features
  --KNN is distance-based, so scaling features is required.

3. Train/Test Split
  --Split the dataset into training and testing sets.

4. Train the KNN Model
  --Example:
    knn = KNeighborsClassifier(n_neighbors=5)

5. Model Evaluation
  --Check:
   --Accuracy
   --Confusion matrix
   --Classification report

   **here is detailes about the model info
   
     --Accuracy: 1.0

    --Confusion Matrix:
     [[10  0  0]
     [ 0 10  0]
     [ 0  0 10]]

    --Classification Report:
                      precision    recall  f1-score   support

        ris-setosa       1.00      1.00      1.00        10
   Iris-versicolor       1.00      1.00      1.00        10
    Iris-virginica       1.00      1.00      1.00        10

         accuracy                           1.00        30
        macro avg       1.00      1.00      1.00        30
     weighted avg       1.00      1.00      1.00        30


6. Tune the K Value
  --Test values of K from 1 to 20 using cross-validation and plot accuracy.
  <img width="585" height="455" alt="KNN_accuracy" src="https://github.com/user-attachments/assets/bda192c9-2ed6-4047-82cd-6e404c9237fd" />

   

7. Visualize Decision Boundaries

  --Using two selected features for easier visualization.
  --Results Summary
  --Best accuracy is usually between K = 5 and K = 7
  --Iris dataset produces high accuracy due to clear class separation
  --Feature scaling greatly improves the performance of KNN
  --Decision boundaries clearly separate the three species
  
  <img width="844" height="547" alt="KNN_decision_boundry" src="https://github.com/user-attachments/assets/7f80714f-1df9-4b2a-a32a-1f61e5e49a2f" />


Project Structure
knn-iris/
│
├── README.md
├── iris.csv
├── knn_notebook.ipynb
└── requirements.txt

requirements.txt
pandas
numpy
matplotlib
scikit-learn
jupyter
