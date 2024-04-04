""" file description"""
# import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.pipeline import FeatureUnion, Pipeline
# import data
data_cleaned = pd.read_csv('../data/processed/cleaned_data.csv')

# import cleaned data
X = data_cleaned.drop(columns=['mcr_DT_output', 'cdu_DT_output', 'isb_DT_output', 'oil_amount_to_recover.1'])
y = data_cleaned[['mcr_DT_output', 'cdu_DT_output', 'isb_DT_output']]

#%% Feature Engineering
# Normalizing numerical var
numerical_features = X._get_numeric_data().columns
#X_engineered_num = MinMaxScaler().fit_transform(X[numerical_features])
#X_engineered_num = pd.DataFrame(X_engineered_num, columns=numerical_features)
#X_engineered_num.index=X.index
numerical_pipeline = Pipeline(steps=[("std_scaler", MinMaxScaler(numerical_features))])


# Catagorical columns
# Encoding categorical var
# Lets drop # seawater & Rtime_ss column; they had only one unique value, & we will use one hot encoder with DROP first.
categorical_features = list(set(X.columns) - set(numerical_features) - set(['displacement.1', 'oil_spill_size', 'Rtime_ss', 'Rtime_sw']))
# displacement.1 is a bug , oil_spill_size will be treated separately

# Treat oil_spill_size column differently
X_engineered_cat = pd.DataFrame()
X_engineered_cat['oil_spill_size'] = X['oil_spill_size'].values
X_engineered_cat['oil_spill_size'].replace(['SMALL','MEDIUM', 'LARGE'], [6, 699, 4999])  # ++ think

# All categorical features (except oil_spill_size) are binary [e.g. yes/no]
# So, applying OneHotEncoder will convert them into 0, 1 with drop=first with drop first category in each feature
# hence, total # of features in processed data wil be the same as original cleaned dataset
encoder = OneHotEncoder(drop='first', handle_unknown="ignore", sparse=False)
X_engineered_cat = pd.DataFrame(encoder.fit_transform(X[categorical_features])) #, columns=cat_cols
X_engineered_cat.index=X.index
X_engineered_cat.columns = categorical_features

categorical_pipeline = Pipeline(steps=[
    ("ohe", OneHotEncoder(
        handle_unknown="ignore",
        sparse=False,
        categories_=categorical_features)
    )
])

# Save the dataset
data_engineered_PLeR = pd.concat([X_engineered, y], axis = 1)
data_engineered_PLeR.to_excel('Inputs/data_engineered_PLeR.xlsx')

feature_pipeline = FeatureUnion(
    n_jobs=-1,
    transformer_list=[
        ("numerical_pipeline", numerical_pipeline),
        ("categorical_pipeline", categorical_pipeline),
    ]
)
"""
"data_engineered_PLeR.xlsx" file contains Y's as categories
e.g. [OK’, ‘Consider’, ‘Go next season’, ‘Unknown’ and ‘Not recommended’]

after saving the data as .csv file, values of Y's are manually converted. <br>
[‘OK’, ‘Consider’, ‘Go next season’, ‘Unknown’ and ‘Not recommended’] are encoded into <br>
[10,   8,           2,               -2,            -10] using Excel's Find & Replace. The file is saved as in SAVE AS "data_engineered_PLeR_Modeling input.xlsx"

"""