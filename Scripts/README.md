### How to train the models?

1- First, make sure you have all necessary packages installed on your environment. In case you encounter issues with the “sickit-bio” package, make sure you upgrade to the latest version using the command:

If you are uisng pip:

``pip install --upgrade scikit-bio``

If you are using conda:

``conda update scikit-bio``

(the version used during experiments is 0.5.5);

2- Second, make sure you add the “Redundancy.r” script and the model dataset in your main folder; 

3- Finally, uncomment one of the five classifiers you want to run starting from line 527 to 531;

4- You are now all set to start training the model!
