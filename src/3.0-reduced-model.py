""" Reduced model"""

# %% -------------- Data Import
# Import libraries
import pandas as pd
# preprocessing
from sklearn.preprocessing import label_binarize, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
# Modeling
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Performance metrics
import joblib
from sklearn import metrics

print(joblib.__version__)
# in terminal ... pip install pipreqnb  , then run pipreqnb
# %%
data = pd.read_excel('../data/processed/cleaned_data.xlsx', index_col='Scene no.').copy()
print(data.columns)
features_10 = ['evaporation_and_natural_disperson', 'E_ss', 'E_sl', 'E_sw', 'sufficient_mixing_energy', 'E_ssC',
               'seawater', 'E_ssI', 'soot_pollution', 'displacement']
X_reduced = data[features_10]
y = data[['mcr_DT_output', 'cdu_DT_output', 'isb_DT_output']]

# Train test Split
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.20, random_state=12)

# Drop y with Consider class which has only 5 records
# y = y[y['CDU options'] != 8]
# X = X.drop(['Scene 26', 'Scene 1247', 'Scene 1380', 'Scene 1655', 'Scene 2109']) # y[y['CDU options'] == 8].index.values

# %% ------------------------ Feature Engineering
numerical_features = tuple(X_reduced._get_numeric_data().columns)
categorical_features = list(set(X_reduced.columns) - set(numerical_features) - set(
    ['displacement.1', 'oil_spill_size', 'Rtime_ss', 'Rtime_sw']))

numerical_transformer = Pipeline(steps=[("MinMaxScaler", MinMaxScaler(feature_range=(1, 2)))])  # numerical_features
categorical_transformer = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])
# ++ need to add spill size convert 4,699,9999

# TRANSFORM NUMERICAL & CATEGORICAL FEATURES SEPARATELY USING ColumnTransformer
feature_pipeline = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)],
    remainder='drop')

# %%  ----------------- Modeling
steps = [("feature_pipeline", feature_pipeline),
         ("model", MultiOutputClassifier(GaussianNB()))]
pipe = Pipeline(steps)
model_BIMReTA = pipe.fit(X_train, y_train)
# Save the reduced model
joblib.dump(model_BIMReTA, '../models/model_BIMReTA2.pkl')
# Classify
y_pred = model_BIMReTA.predict(X_test)
print(X_test.shape)
y_score = model_BIMReTA.predict_proba(X_test)
print(y_pred)

# %%  Model Assessment
print('----------------------------Confusion Matrix--------------')
print('MCR', metrics.confusion_matrix(y_test.iloc[:, 0], y_pred[:, 0]))
print('CDU', metrics.confusion_matrix(y_test.iloc[:, 1], y_pred[:, 1]))
print('ISB', metrics.confusion_matrix(y_test.iloc[:, 2], y_pred[:, 2]))
print('\n----------------------------Classification Report--------------')
print('MCR', metrics.classification_report(y_test.iloc[:, 0], y_pred[:, 0]))
print('CDU', metrics.classification_report(y_test.iloc[:, 1], y_pred[:, 1]))
print('ISB', metrics.classification_report(y_test.iloc[:, 2], y_pred[:, 2]))
print('\n----------------------------ROC AUC--------------')
print('MCR', metrics.roc_auc_score(y_test.iloc[:, 0], y_score[0], multi_class='ovo'))
print('CDU', metrics.roc_auc_score(y_test.iloc[:, 1], y_score[1], multi_class='ovo'))
print('ISB', metrics.roc_auc_score(y_test.iloc[:, 2], y_score[2], multi_class='ovo'))

# %%----------------------------  Compare with Full model
# The comparison is conducted on engineered data for simplification
data_engineered = pd.read_excel('../data/processed/engineered_data.xlsx', index_col='Scene no.').copy()
X = data_engineered[features_10]
print(data_engineered.columns)
y = data_engineered[['MCR options', 'CDU options', 'ISB options']]
# Data for reduced model
X_reduced_mcr = X[['evaporation_and_natural_disperson', 'E_ss', 'E_sl', 'E_sw']]
y_m_b = label_binarize(y['MCR options'], classes=[10, 8, 2, -2])
X_train_mcr, X_test_mcr, y_train_mcr, y_test_mcr = train_test_split(X_reduced_mcr, y_m_b, test_size=0.20,
                                                                    random_state=12)
X_reduced_cdu = X[['sufficient_mixing_energy', 'E_ssC', 'seawater']]
y_c_b = label_binarize(y['CDU options'], classes=[10, 8, -2, -10])
X_train_cdu, X_test_cdu, y_train_cdu, y_test_cdu = train_test_split(X_reduced_cdu, y_c_b, test_size=0.20,
                                                                    random_state=12)
X_reduced_isb = X[['E_ssI', 'soot_pollution', 'displacement']]
y_i_b = label_binarize(y['ISB options'], classes=[10, 8, -2])
X_train_isb, X_test_isb, y_train_isb, y_test_isb = train_test_split(X_reduced_isb, y_i_b, test_size=0.20,
                                                                    random_state=12)

# %%---------------------------------------------------------- Compare Algorithms
seed = 7
models = [('NB', OneVsRestClassifier(GaussianNB()))]
results = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results_m = model_selection.cross_val_score(model, X_reduced_mcr, y_m_b, cv=kfold, scoring='f1_samples')
    results.append(cv_results_m)
results.append(model_selection.cross_val_score(model, X, y_m_b, cv=kfold, scoring='roc_auc'))
results.append(model_selection.cross_val_score(model, X, y_c_b, cv=kfold, scoring='roc_auc'))
results.append(model_selection.cross_val_score(model, X, y_i_b, cv=kfold, scoring='roc_auc'))
results.append(model_selection.cross_val_score(model, X_reduced_mcr, y_m_b, cv=kfold, scoring='roc_auc'))
results.append(model_selection.cross_val_score(model, X_reduced_cdu, y_c_b, cv=kfold, scoring='roc_auc'))
results.append(model_selection.cross_val_score(model, X_reduced_isb, y_i_b, cv=kfold, scoring='roc_auc'))

# %%
boxplot_df = pd.DataFrame(results).T
fig4, ax = plt.subplots(figsize=(8, 10))
sns.boxplot(data=boxplot_df, boxprops=dict(alpha=.3))
fig4.savefig('../reports/Fig8 boxplot.png', dpi=600)
boxplot_df.mean(axis=0)
plt.show()

# %%
### ROC Curve of reduced model
target_names = ['MCR options', 'CDU options', 'ISB options']
for target_name in target_names:
    if target_name == 'MCR options':
        X_r = X[['evaporation_and_natural_disperson', 'E_ss', 'E_sl', 'E_sw']]
        classes = [10, 8, 2, -2]
        class_m = ['OK', 'Consider', 'Go next season', 'Unknown'];
        color_m = ['green', 'limegreen', 'blue', 'lightgray']
    elif target_name == 'CDU options':
        X_r = X[['sufficient_mixing_energy', 'E_ssC', 'seawater']]
        classes = [10, 8, -2, -10]
        class_m = ['OK', 'Consider', 'Unknown', 'Not recommended'];
        color_m = ['green', 'limegreen', 'blue', 'lightgray']
    else:
        X_r = X[['E_ssI', 'soot_pollution', 'displacement']]
        classes = [10, 8, -2]
        class_m = ['OK', 'Consider', 'Unknown'];
        color_m = ['green', 'limegreen', 'darkgray']

    y_b = label_binarize(y[target_name], classes=classes)
    n_classes = y.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X_r, y_b, test_size=0.20, random_state=12)
    classifier = OneVsRestClassifier(GaussianNB()).fit(X_train, y_train)
    y_score = classifier.predict_proba(X_test)
    y_pred = classifier.predict(X_test)
    # Calculating metrics
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(y_test.shape[1]):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Produce the figures
    fig5 = plt.figure()
    for i, color in zip(range(y_test.shape[1]), color_m):
        plt.plot(fpr[i], tpr[i], color=color,
                 label=class_m[i])
    plt.plot([0, 1], [0, 1], '--', color='lightgray', lw=1)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
    fig5.savefig(f'../reports/ROC curve {target_name}.png', dpi=600)

# %% ---------------- ## Hyperparameter tuning (improve the model performance)
parameters = {'estimator__var_smoothing': [1e-11, 1e-10, 1e-9]}
Bayes = GridSearchCV(classifier, parameters, scoring='roc_auc', cv=10).fit(X_train_mcr, y_train_mcr)
print(Bayes.best_estimator_)
print('best score:')
print(Bayes.best_score_)
y_score_mcr = Bayes.best_estimator_.predict(X_test_mcr)

fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(y_test_mcr.shape[1]):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test_mcr[:, i], y_score_mcr[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
