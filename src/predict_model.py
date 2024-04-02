import pandas as pd
import joblib


# load the trained model (BIMReTA)
model_gnb_multioutput = joblib.load('Outputs\model_gnb_multioutput.pkl')

# predict on holdout set (when no data is passed)
pred_holdout = predict_model(catboost)

# predict on new dataset
new_data = pd.read_csv('new-data.csv')
pred_new = predict_model(catboost, data = new_data)



