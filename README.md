# Bayesian Inference Modeling to Rank Cleanup System in Arctic oil spill

# Summary
**Situation**: 
Selecting the correct response technology for emergency oil spill response is difficult. This work will rank response technologies (from MCR, CDU, & ISB) considering Arctic environement.  


**Task**:
1. Since Oil Spill is rare in Arctic, we needed to build **data pipeline** to obtain reasonable numbers of incidents
2. **Build model** to classify technologies


**Action**:
1. Based on Monte Carlo Simulation (implemented using distribution of feature variables, and outputs obtained from engineering model), 3100 scenarios is generated
2. A #multi-class, multi-label classification system is developed. Bayesian Inference model is implemented using Naive Bayes Classifier. Multi-label: y = [y1, y2, y3] = [MCR, CDU, ISB]. Each label can have multiple classes e.g. [OK, Consider, Go Next Season, Unknown, Not recommended]


**Result**: 
Based on oil and environmental conditions in Arctic, our model proposes which technology would be better to respond oil spill. The model has 0.79, 0.93 and 0.93 ROC-AUC score for different technologies.


# Directory
    BIM
    ├── requirement.txt         Dependencies
    ├── README.md               Project README
    ├── data                    
    │   ├── raw                 Raw files e.g. 
    │   ├── processed           Cleaned and processed data
    ├── models                  
    │   └── model_BIMReTA.pkl   Trained models 
    │   ├── ...
    ├── reports                 Figures in the paper
    │   ├── Fig3                Bar plot
    │   ├── Fig5a.png           ROC curve
    └── src                     Source files
        ├── 2.0-bayesian-model.py
        ├── ...
        └── 5.0-nn.py
    ├── BIMReTA_app.py          Streamlit web app
    ├── Dockerfile    


The picture below is an overview of the **project's methodology**. Further details can be found in [this journal paper](https://doi.org/10.1016/j.marpolbul.2022.114203).
![image](https://user-images.githubusercontent.com/19787712/220946219-a9e7b486-3a50-491f-8182-92630f2b04d9.png)

# How to cite
Tanmoy Das, Floris Goerlandt (2022). Bayesian inference modeling to rank response technologies in arctic marine oil spills. Marine Pollution Bulletin, 185, 114203. https://doi.org/10.1016/j.marpolbul.2022.114203



