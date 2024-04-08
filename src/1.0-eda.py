""" File Name: EDA & Engineering.ipynb <br>
Developer: Tanmoy Das <br>
Date: July 01, 2022 (Revised April 2024) <br>

Outline:
1. Data Import & basic cleaning
2. EDA
3. Feature Engineering """

# %% -------------------------------------------- Import data --------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Import data
data = pd.read_excel('../data/processed/cleaned_data.xlsx').copy()
data.info()
print(data.tail(5))

# %% -------------------------------------------- Data Exploration --------------------------------------------
data.describe(include='all') 
print(data['E_ss'].value_counts())
print(data['E_sl'].value_counts())
print(data['E_sw'].value_counts())
print(data['sufficient_mixing_energy'].value_counts())
print(data['E_ssC'].value_counts())
print(data['seawater'].value_counts())
print(data['E_ssI'].value_counts())
print(data['soot_pollution'].value_counts())
print(data['displacement'].value_counts())

data.describe(include=[object])
data.columns

# -------------------------------------------- Properties of categorical variables
df_ = data.select_dtypes(exclude=['int', 'float'])
for col in df_.columns:
    print(df_[col].unique()) 
    print(df_[col].value_counts()) 
# -------------------------------------------- Remove duplicate values
print(data.duplicated().sum())
print(data.duplicated().value_counts())

# -------------------------------------------- Give name to 1st column and add text to INDEX like Scene 1, Scene 2
data = data.rename(columns={"Sr no.": "Scene no."})
data['Scene no.'] = data['Scene no.'] # change value
data['Scene no.'] ='Scene ' + data['Scene no.'].astype(str)
# Index this column to the dataframe
data = data.set_index('Scene no.')

# %%
# Count values in target variables
print('MCR options: unique value count')
print(data['mcr_DT_output'].value_counts())
print('\nCDU options: unique value count')
print(data['cdu_DT_output'].value_counts())
print('\nISB options: unique value count')
print(data['isb_DT_output'].value_counts())

# %% -------------------------------------------- Data Cleaning --------------------------------------------
data_cleaned = data.copy()
data_cleaned['mcr_DT_output'] = data_cleaned['mcr_DT_output'].map(lambda x: x.strip('[]'))
data_cleaned['mcr_DT_output'] = data_cleaned['mcr_DT_output'].str.replace('\'', '')
data_cleaned['mcr_DT_output'] = data_cleaned['mcr_DT_output'].str.replace('\d+', '')
data_cleaned['mcr_DT_output'] = data_cleaned['mcr_DT_output'].map(lambda x: x.strip('[]')) 
data_cleaned['cdu_DT_output'] = data_cleaned['cdu_DT_output'].str.replace('\'', '')
data_cleaned['cdu_DT_output'] = data_cleaned['cdu_DT_output'].str.replace('\d+', '')
data_cleaned['mcr_DT_output'] = data_cleaned['mcr_DT_output'].map(lambda x: x.strip('[]')) 
data_cleaned['isb_DT_output'] = data_cleaned['isb_DT_output'].str.replace('\'', '')
data_cleaned['isb_DT_output'] = data_cleaned['isb_DT_output'].str.replace('\d+', '')
# convert [OK, consider] into OK 
data_cleaned['isb_DT_output'] = data_cleaned['isb_DT_output'].str.split(',').str[0] 
data_cleaned['mcr_DT_output'] = data_cleaned['mcr_DT_output'].str.replace("ok","OK")

# remove unwanted characters e.g. [, ],
data_cleaned['cdu_DT_output'] = data_cleaned['cdu_DT_output'].str.replace("]","")
data_cleaned['cdu_DT_output'] = data_cleaned['cdu_DT_output'].str.replace("[","")

data_cleaned['isb_DT_output'] = data_cleaned['isb_DT_output'].str.replace("]","")
data_cleaned['isb_DT_output'] = data_cleaned['isb_DT_output'].str.replace("[","")
data_cleaned[['cdu_DT_output','isb_DT_output']]

data_cleaned['cdu_DT_output'] = data_cleaned['cdu_DT_output'].str.replace("]","")

# -------------------------------------------- Replace empty values with UNKNOWN
data_cleaned[['mcr_DT_output', 'cdu_DT_output', 'isb_DT_output']] = data_cleaned[['mcr_DT_output', 'cdu_DT_output', 'isb_DT_output']] .replace('','Unknown') # Specific columns

print('MCR options: unique value count')
print(data_cleaned['mcr_DT_output'].value_counts())
print('\nCDU options: unique value count')
print(data_cleaned['cdu_DT_output'].value_counts())
print('\nISB options: unique value count')
print(data_cleaned['isb_DT_output'].value_counts())

# -------------------------------------------- # EDA
# create a dataset
height_mcr = [1477, 345, 131, 1147]
bars_mcr = ('OK', 'Consider', 'Go to next season', 'Unknown')
x_pos_mcr = np.arange(len(bars_mcr))
height_cdu = [933, 5, 2042, 120]
bars_cdu = ('OK', 'Consider', 'Not recommended', 'Unknown')
x_pos_cdu = np.arange(len(bars_cdu))
height_isb = [2098, 846, 156, 0]
bars_isb = ('OK', 'Consider', 'Unknown', '')
x_pos_isb = np.arange(len(bars_isb))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.bar(x_pos_mcr, height_mcr, color=['green', 'limegreen', 'blue', 'gray'], alpha = .5)
ax2.bar(x_pos_cdu, height_cdu, color=['green', 'palegreen', 'red', 'gray'], alpha = .5)
ax3.bar(x_pos_isb, height_isb, color=['green', 'limegreen',  'gray', 'white'], alpha = .5)
ax1.set_yticks(np.arange(0, 2800, 400))
ax2.set_yticks(np.arange(0, 2800, 400))
ax3.set_yticks(np.arange(0, 2800, 400))
ax1.axes.xaxis.set_ticklabels([])
ax1.tick_params(bottom=False)
ax2.axes.yaxis.set_ticklabels([]) # remove y-tick of ax2
ax3.axes.yaxis.set_ticklabels([]) 
ax2.axis('off'); ax3.axis('off') #
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False); ax1.spines['bottom'].set_visible(False)
plt.savefig('../reports/distribution of Ys.svg', 
                          dpi=600, facecolor='w', edgecolor='b',
                          orientation='portrait', format=None, transparent=False,
                          bbox_inches=None,  pad_inches=0, metadata=None)
plt.show()

# %% --------------------------------------------
sns.catplot(x="mcr_DT_output", kind="count", palette="ch:.25", data=data_cleaned)
plt.show()

# Grouped bar chart with 3 different categorical columns
# Preparing data for Histogram
subclasses = ['OK', 'Consider', 'Go to next season', 'Unknown', 'Not Recommended']
response_systems = ['MCR', 'CDU', 'ISB']

# -------------------------------------------- Bubbleplot
plt.figure(figsize=(8, 6), dpi=300)
fig = px.scatter(data_cleaned, x="persistence", y="evaporation_and_natural_disperson", animation_frame="mcr_DT_output",
           size="oil_amount_to_recover",  color="seasurface" ) 

# fig["layout"].pop("updatemenus") # optional, drop animation buttons
fig.update_layout(template="none")
fig.show()

# -------------------------------------------- Strip plot
sns.swarmplot(x="mcr_DT_output", y="evaporation_and_natural_disperson", data=data_cleaned)
plt.show()

sns.stripplot(x="cdu_DT_output", y="E_ssC", hue='sufficient_mixing_energy', data=data_cleaned)
plt.show()

sns.catplot(x="mcr_DT_output", y="evaporation_and_natural_disperson",hue="displacement", kind="point", data=data_cleaned,
           linestyles=["-", "--"],  markers=["^", "o"],  palette={"no": "lightgrey", "yes": "b"})
plt.show()

# -------------------------------------------- KDE Plot
sns.kdeplot(data=data_cleaned, x="oil_amount_to_recover", hue="mcr_DT_output")
plt.show()
