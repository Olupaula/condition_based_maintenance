# Condition Base Maintenance of Naval Propulsion Plants
This project titled [Condition Based Maintenance](https://en.wikipedia.org/wiki/Predictive_maintenance) 
of [Naval Propulsion Plants](https://en.wikipedia.org/wiki/Nuclear_marine_propulsion) deals with experiments were conducted using a numerical simulator of a naval 
vessel (Frigate) characterized by a Gas Turbine (GT) propulsion plant with the aim of 
determining the performance decay or degradation of the vessel. Condition Base Maintenance is also known as
predictive maintenance. 
    Three measures of the performance decay of the vessels considered were: Ship speed (linear function of 
the lever position lp), Compressor degradation coefficient kMc and Turbine degradation coefficient kMt.


It makes good sense to develop a
[model](https://learn.microsoft.com/en-us/windows/ai/windows-ml/what-is-a-machine-learning-model) 
capable of predicting the performance decay of any given vessel, as a predictive maintenance which is a form of 
preventive maintenance is usually more cost-effective than corrective maintenance. 
[See](https://roadtoreliability.com/types-of-maintenance/)

In this project, a number of [regression models](https://learn.microsoft.com/en-us/training/modules/understand-regression-machine-learning/) 
were considered to see which model will be the best for predicting the performance decay of naval propulsion plants.

### Data Source: [Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants)

### Data Features:

1. Gas Turbine shaft torque (GTT) [kN m]
2. Gas Turbine rate of revolutions (GTn) [rpm]
3. Gas Generator rate of revolutions (GGn) [rpm]
4. Starboard Propeller Torque (Ts) [kN]
5. Port Propeller Torque (Tp) [kN]
6. HP Turbine exit temperature (T48) [C]
7. GT Compressor inlet air temperature (T1) [C]
8. GT Compressor outlet air temperature (T2) [C]
9. HP Turbine exit pressure (P48) [bar]
10. GT Compressor inlet air pressure (P1) [bar]
11. GT Compressor outlet air pressure (P2) [bar]
12. Gas Turbine exhaust gas pressure (Pexh) [bar]
13. Turbine Injecton Control (TIC) [%]
14. Fuel flow (mf) [kg/s]

### Selected (Utilized) Features:
1. Gas Turbine shaft torque (GTT) [kN m]
2. Gas Generator rate of revolutions (GGn) [rpm]

### Data Targets:
1. Ship speed (v) [knots] (a linear function of Lever position (lp) )
2. GT Compressor decay state coefficient.
3. GT Turbine decay state coefficient

### Feature Selection Method Used:
The [Variance Inflation Factor (VIF)](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwj-rvWLx5v_AhWJ_rsIHemtBwQQFnoECA4QAQ&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FVariance_inflation_factor&usg=AOvVaw39FcOQct2OEPZVWf72UUez) was used to check for the presence of multicollinearity between the features. 
It was found that when considering all the features, they had very high VIF's which were greater than traditional 
benchmark of a VIF of 5 or below and above the permissible benchmark of a VIF 10 or below (suggest by some scholars).
When considering three of more features, they had high VIF's still. Hence, only two features were selected because their
VIF's were less than 5.

### Data Visualization:
<p>
    <img src="./cbm_images/Gas Generator rate of revolutions (GGn) [rpm]_Ship_speed_(v)_[knots].png">
    <p>
        The relationship between Gas Generator rate of revolutions and ship speed is a curve, which means that a linear
        model is not suitable.
    </p>
</p>
<p> 
    <img src="./cbm_images/Gas Turbine shaft torque (GTT) [kN m]_Ship_speed_(v)_[knots].png">
        The relationship between Gase Turbine shaft torgue and ship speed is a curve, which means that a linear
        model is not suitable.
    </p>
    </p>
</p>


### Regression Techniques used:
1. [Linear Regression](https://www.oxfordreference.com/display/10.1093/oi/authority.20110803100107226;jsessionid=BAD370C49344F63EAF545090E2E032DE)
2. [K-Nearest Neighbor (KNN)](https://online.stat.psu.edu/stat508/lesson/k)
3. [Support Vector Machine (SVM)](https://www.researchgate.net/publication/221621494_Support_Vector_Machines_Theory_and_Applications/link/0912f50fd2564392c6000000/download)
4. [Decision Tree (DT)](https://online.stat.psu.edu/stat857/node/236/)

### Evaluation Metrics: 
1. [Coefficient of Determination](https://www.oxfordreference.com/display/10.1093/oi/authority.20110803095621787#:~:text=In%20statistics%2C%20a%20measure%20of,Symbol%3A%20R2.)
2. [Mean Squared Error](https://statisticsbyjim.com/regression/mean-squared-error-mse/#:~:text=The%20calculations%20for%20the%20mean,by%20the%20number%20of%20observations.)

### The best Model:
When the using the metric Mean Squared Error, the model with the lowest mean squared model is the best among all models 
under consideration. However, while using the Coefficient of Determination, the model with the highest Coefficient of 
Variation is preferable. Using both metrics, the Decision Tree came up as the best model

[View Code on Kaggle](https://www.kaggle.com/code/oluade111/condition-based-maintenance-of-naval-propulsion)

[Use API]()


