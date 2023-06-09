# Replication Package


This is the replication package for the paper intitled:

__*"What Constitutes the Deployment and Run-time Configuration System? An Empirical Study on OpenStack Projects".*__

This paper investigates the configuration file types that constitute the configuration system of OpenStack. We conduct our study on OpenStack as it is one of the most deployed cloud platform for infrastructure management. To identify the different configruation files, we first manually investigate the different configuration types, where we identified 9 different types of configuration files. Then, we comapre between five classifiers (SVC, RF, LR, KNN, and GB), and leverage two machine learning models to automate the identification of the different configruation file types. The first model predicts the configuration from the non-configuration files. The second model predicts the different types of the configuration files.


### Whats inside the package?

In this repository, we provide: 

1) the datasets of training the [``first model``](https://github.com/stilab-ets/CongIdentification/blob/main/Datasets/Dataset-Model1(configNonconfig).csv) and the [``second model``](https://github.com/stilab-ets/CongIdentification/blob/main/Datasets/Dataset-Model2(ConfigTypes).csv). We also provide the links to the files of the [``first model``](https://github.com/stilab-ets/CongIdentification/blob/main/Datasets/Links-Model1(configNonconfig).csv) and the second model [``second model``](https://github.com/stilab-ets/CongIdentification/blob/main/Datasets/Links-Model2(ConfigTypes).csv)
2) the scripts for building our [``first model``](https://github.com/stilab-ets/CongIdentification/blob/main/Scripts/Script-Model1(configNonconfig).py) and [``second model``](https://github.com/stilab-ets/CongIdentification/blob/main/Scripts/Script-Model2(ConfigTypes).py).   
3) the comparison results between the five classifiers for the [``first model``](https://github.com/stilab-ets/CongIdentification/blob/main/Classifiers-Results/Results-Model1(configNonconfig).csv) and the [``second model``](https://github.com/stilab-ets/CongIdentification/blob/main/Classifiers-Results/Results-Model2(ConfigTypes).csv).


### How to replicate the study?









