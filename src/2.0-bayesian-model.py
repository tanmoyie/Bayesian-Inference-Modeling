"""Bayesian Inference (Full Model) <br>
Developer: Tanmoy Das <br>
Date: June 13, 2022 (Revision: April 2024)

Outline:
- Data processing <br>
- Modeling <br>
- Model Assessment and Validation <br>
- Model Selection """

# Import required Python libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import joblib

#%% --------------------------------------------------- Modeling -------------------------------------------------------
# import data
engineered_data = pd.read_excel("../data/processed/engineered_data.xlsx", index_col='Scene no.').copy()
X = engineered_data.drop(columns=['MCR options', 'CDU options', 'ISB options'])
y = engineered_data[['MCR options', 'CDU options', 'ISB options']]
# Drop y with Consider class which has only 5 records
y = y[y['CDU options'] != 8]
X = X.drop(['Scene 26', 'Scene 1247', 'Scene 1380', 'Scene 1655', 'Scene 2109']) # y[y['CDU options'] == 8].index.values
# Data Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12)

# modeling
model_GB_ins = GaussianNB()
model_gnb_multioutput = MultiOutputClassifier(model_GB_ins).fit(X_train, y_train)

# Save the reduced model
joblib.dump(model_gnb_multioutput, '../models/full_model_BIMReTA.pkl')
# predict
y_pred = model_gnb_multioutput.predict(X_test)
y_score = model_gnb_multioutput.predict_proba(X_test)


#%%
print('----------------------------Confusion Matrix--------------')
print('MCR', metrics.confusion_matrix(y_test.iloc[:,0], y_pred[:,0]))
print('CDU', metrics.confusion_matrix(y_test.iloc[:,1], y_pred[:,1]))
print('ISB', metrics.confusion_matrix(y_test.iloc[:,2], y_pred[:,2]))

print('\n----------------------------Classification Report--------------')
print('MCR', metrics.classification_report(y_test.iloc[:,0],y_pred[:,0]))
print('CDU', metrics.classification_report(y_test.iloc[:,1],y_pred[:,1]))
print('ISB', metrics.classification_report(y_test.iloc[:,2],y_pred[:,2]))

print('\n----------------------------ROC AUC--------------')
print('MCR', metrics.roc_auc_score(y_test.iloc[:,0],y_score[0], multi_class='ovo'))
print('CDU', metrics.roc_auc_score(y_test.iloc[:,1],y_score[1], multi_class='ovo'))
print('ISB', metrics.roc_auc_score(y_test.iloc[:,2],y_score[2], multi_class='ovo'))

#%% Figure: ROC curve
target_names = ['MCR options', 'CDU options', 'ISB options']
for target_name in target_names:
    if target_name == 'MCR options':
        classes = [10, 8, 2, -2]
        class_m = ['OK', 'Consider', 'Go next season', 'Unknown']
        color_m = ['green', 'limegreen', 'blue', 'lightgray']
    elif target_name == 'CDU options':
        classes = [10, 8, -2, -10]
        class_m = ['OK', 'Consider', 'Unknown', 'Not recommended']
        color_m = ['green', 'limegreen', 'blue', 'lightgray']
    else:
        classes = [10, 8, -2]
        class_m = ['OK', 'Consider', 'Unknown']
        color_m = ['green', 'limegreen',  'darkgray']

    y_b = label_binarize(y[target_name], classes=classes)
    n_classes = y.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y_b, test_size=0.20, random_state=12)
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
#  ipynb-py-convert notebooks/4.0-xplainable-ai.ipynb src/4.0-xplainable-ai.py  run in Terminal


#%% ----------------------------------------- Neural Network
# ++