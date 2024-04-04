""" This file will fetch data from app UI, and make predictions"""
import pandas as pd
import joblib
import build_features

#%% --------------------------- Data & Model ---------------------------
# Fetch data from UI
# ++
# data =

#

# load the trained model (BIMReTA)
model_BIMReTA = joblib.load('models\mmodel_BIMReTA.pkl')
# pipeline combines model and data

#%% --------------------------- Prediction ---------------------------
# Predict on new dataset
# The only required code line to make the prediction using our pipeline
y_pred = survival_pipe.predict(X_valid)

# Show the result
print(y_pred)

