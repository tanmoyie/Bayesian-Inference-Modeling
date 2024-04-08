""" Xplainable AI : Analyze previously build models & visualize them.
Outline:
1. Import model & basic statistics
2. Visualization: Summary ploy, prediction probability, SHAP values
Developer: Tanmoy Das,
Date: May 25, 2022"""

# %%
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from lime.lime_tabular import LimeTabularExplainer
import joblib

# import data
engineered_data = pd.read_excel("../data/processed/engineered_data.xlsx", index_col='Scene no.').copy()
X = engineered_data.drop(columns=['MCR options', 'CDU options', 'ISB options'])
y = engineered_data[['MCR options', 'CDU options', 'ISB options']]
# Drop y with Consider class which has only 5 records
y = y[y['CDU options'] != 8]
X = X.drop(['Scene 26', 'Scene 1247', 'Scene 1380', 'Scene 1655', 'Scene 2109']) # y[y['CDU options'] == 8].index.values
# Data Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12)

# %% ----------------------------------------------- Load the model from the file
model_gnb_multioutput = joblib.load('../models/full_model_BIMReTA.pkl')
y_pred = model_gnb_multioutput.predict(X_test)

# %% ----------------------------------------------- Model building
model_gnb_mcr = GaussianNB().fit(X_train, y_train['MCR options'])
model_gnb_cdu = GaussianNB().fit(X_train, y_train['CDU options'])
model_gnb_isb = GaussianNB().fit(X_train, y_train['ISB options'])

# ----------------------------------------------- Visualization of full model
row_no = 3
explainer_mcr = LimeTabularExplainer(X_train.values, feature_names=X_train.columns,
                                 class_names=[10, 8, 2, -2], discretize_continuous=True, kernel_width=5)
exp_mcr = explainer_mcr.explain_instance(X_train.values[row_no], model_gnb_mcr.predict_proba, num_features=10)
exp_mcr.show_in_notebook(show_table=True, show_all=True)

explainer_cdu = LimeTabularExplainer(X_train.values, feature_names=X_train.columns,
                                 class_names=[10, 8, -2, -10], discretize_continuous=True, kernel_width=5)
exp_cdu = explainer_cdu.explain_instance(X_train.values[row_no], model_gnb_cdu.predict_proba, num_features=10)
exp_cdu.show_in_notebook(show_table=True, show_all=False)

explainer_isb = LimeTabularExplainer(X_train.values, feature_names=X_train.columns,
                                 class_names=[10, 8, -2], discretize_continuous=True, kernel_width=5)
exp_isb = explainer_isb.explain_instance(X_train.values[row_no], model_gnb_isb.predict_proba, num_features=10)
exp_isb.save_to_file('../reports/Fig04.html', show_table=True, show_all=False)
