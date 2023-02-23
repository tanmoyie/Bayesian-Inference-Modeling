# Bayesian Inference Modeling
## Situation
Selecting correct response technology for emergency oil spill response is difficult. 
A multi-class, multi-label classification system is developed, and Bayesian Inference Modeling is applied.

## Task
1. Since Oil Spill is rare in Arctic, we needed to build **data pipeline** to obtain reasonable numbers of incidents
2. **Build model** to classify technologies


## Action
1. Based on Monte Carlo Simulation (implemented using distribution of feature variables, and outputs obtained from engineering model), 3100 scenarios is generated
2. Bayesian Inference model is implemented using Naive Bayes Classifier


## Result 
Based on oil and environmental conditions in Arctic, our model proposes which technology would be better to respond oil spill.

![image](https://user-images.githubusercontent.com/19787712/220946219-a9e7b486-3a50-491f-8182-92630f2b04d9.png)
Further details can be found in [this journal paper]([url](https://doi.org/10.1016/j.marpolbul.2022.114203))

