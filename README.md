# CongIdentification


This is the Github assets folder for the paper intitled "What Constitute a Configuration System? An Empirical Study on OpenStack Projects".
This paper investigates the configuration file types that constitute the configuration system of OpenStack. We conduct our study on OpenStack as it is one of the most deployed cloud platform for infrastructure management. To identify the different configruation files, we comapre between five classifiers (SVC, RF, LR, KNN, and GB), and leverage two machine learning models. The first model predicts configuration from non-configuration files. The second model predicts the different types of the configuration files.

In this repository, we provide: 

1) the datasets of training the [``first model``](https://github.com/Narjes-b/CongIdentification/blob/main/Datasets/Model1(configNonconfig).csv) and the [``second model``](https://github.com/Narjes-b/CongIdentification/blob/main/Datasets/Model2(ConfigTypes).csv). 
2) the scripts for building our [``first model``](https://github.com/Narjes-b/CongIdentification/blob/main/Scripts/Model1(configNonconfig).py) and [``second model``](https://github.com/Narjes-b/CongIdentification/blob/main/Scripts/Model2(ConfigTypes).py).   
3) The comparison results between the five classifiers for the [``first model``](https://github.com/Narjes-b/CongIdentification/blob/main/Classifiers-Results/Model1(configNonconfig).csv) and the [``second model``](https://github.com/Narjes-b/CongIdentification/blob/main/Classifiers-Results/Model2(ConfigTypes).csv).
