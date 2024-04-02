""" file description"""
# import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# import data
data_cleaned = pd.read_csv('../data/processed/cleaned_data.csv')

# import cleaned data
X = data_cleaned.drop(columns=['mcr_DT_output', 'cdu_DT_output', 'isb_DT_output', 'oil_amount_to_recover.1'])
y = data_cleaned[['mcr_DT_output', 'cdu_DT_output', 'isb_DT_output']]

#%% Feature Engineering
# Normalizing numerical var
num_cols = X._get_numeric_data().columns


X_engineered_num = MinMaxScaler().fit_transform(X[num_cols])
X_engineered_num = pd.DataFrame(X_engineered_num, columns=num_cols)
X_engineered_num.index=X.index

# Catagorical columns
# Encoding categorical var
# Lets drop # seawater & Rtime_ss column; they had only one unique value, & we will use one hot encoder with DROP first.
cat_cols = list(set(X.columns) - set(num_cols) - set(['displacement.1', 'oil_spill_size', 'Rtime_ss', 'Rtime_sw'])) # displacement.1 is a bug , oil_spill_size will be treated separately


# Treat oil_spill_size column differently
X_engineered_cat = pd.DataFrame()
X_engineered_cat['oil_spill_size'] = X['oil_spill_size'].values
X_engineered_cat['oil_spill_size'].replace(['SMALL','MEDIUM', 'LARGE'], [6, 699, 4999])  # ++ think

encoder = OneHotEncoder(drop = 'first', handle_unknown = "ignore", sparse = False)
X_engineered_cat = pd.DataFrame(encoder.fit_transform(X[cat_cols])) #, columns=cat_cols
X_engineered_cat.index=X.index

X_engineered_cat.columns = cat_cols


# combine categorical & numerical var
X_engineered = pd.concat([X_engineered_num, X_engineered_cat], axis = 1)
X_engineered = pd.DataFrame(X_engineered, index=X.index) # (index=df1.index)


print('MCR options: unique value count')
display(y['mcr_DT_output'].value_counts())
print('\nCDU options: unique value count')
display(y['cdu_DT_output'].value_counts())
print('\nISB options: unique value count')
display(y['isb_DT_output'].value_counts())

# Save the dataset
data_engineered_PLeR = pd.concat([X_engineered, y], axis = 1)
data_engineered_PLeR.to_excel('Inputs/data_engineered_PLeR.xlsx')

"""
"data_engineered_PLeR.xlsx" file contains Y's as categories
e.g. [OK’, ‘Consider’, ‘Go next season’, ‘Unknown’ and ‘Not recommended’]

after saving the data as .csv file, values of Y's are manually converted. <br>
[‘OK’, ‘Consider’, ‘Go next season’, ‘Unknown’ and ‘Not recommended’] are encoded into <br>
[10,   8,           2,               -2,            -10] using Excel's Find & Replace. The file is saved as in SAVE AS "data_engineered_PLeR_Modeling input.xlsx"

"""