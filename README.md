# CongIdentification


This is the replication package for the paper intitled "What Constitutes a Configuration System? An Empirical Study on OpenStack Projects".
This paper investigates the configuration file types that constitute the configuration system of OpenStack. We conduct our study on OpenStack as it is one of the most deployed cloud platform for infrastructure management. To identify the different configruation files, we first manually investigate the different configuration types. Then, we comapre between five classifiers (SVC, RF, LR, KNN, and GB), and leverage two machine learning models to automate the identification of these types. The first model predicts configuration from non-configuration files. The second model predicts the different types of the configuration files.

In this repository, we provide: 

1) the datasets of training the [``first model``](https://github.com/Narjes-b/CongIdentification/blob/main/Datasets/Dataset-Model1(configNonconfig).csv) and the [``second model``](https://github.com/Narjes-b/CongIdentification/blob/main/Datasets/Dataset-Model2(ConfigTypes).csv). 
2) the scripts for building our [``first model``](https://github.com/Narjes-b/CongIdentification/blob/main/Scripts/Script-Model1(configNonconfig).py) and [``second model``](https://github.com/Narjes-b/CongIdentification/blob/main/Scripts/Script-Model2(ConfigTypes).py).   
3) The comparison results between the five classifiers for the [``first model``](https://github.com/Narjes-b/CongIdentification/blob/main/Classifiers-Results/Results-Model1(ConfigTypes).csv) and the [``second model``](https://github.com/Narjes-b/CongIdentification/blob/main/Classifiers-Results/Results-Model2(ConfigTypes).csv).


