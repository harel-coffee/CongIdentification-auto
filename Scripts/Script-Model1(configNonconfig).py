## for data
## for interpretation
!pip install lime
import lime
from lime import lime_tabular
# for explainer show_table parameter

#!pip install --upgrade prompt-toolkit==2.0.2 --force-reinstall

import json
from cmath import sqrt
from collections import Counter
from bs4 import BeautifulSoup
from IPython.display import display
import pandas as pd
from matplotlib import pyplot
from numpy import argmax
from tabulate import tabulate
import numpy as np
import scipy as sp
!pip install sparse
from scipy.sparse import csr_matrix
!pip install scikit-bio==0.5.5
import skbio
import rpy2.robjects as R
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)

# Install packages
packnames = ('Hmisc', 'rms','TeachingDemos')
utils.install_packages(StrVector(packnames))

# Load packages
hmisc = importr('Hmisc')
TeachingDemos = importr('TeachingDemos')
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter



## for plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
!pip install eli5
import warnings
warnings.filterwarnings('ignore')
import eli5
import string
from eli5.sklearn import PermutationImportance
## for processing
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
import pickle
from numpy import array 
## for bag-of-words
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing, \
   feature_selection, metrics, svm
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
from sklearn.feature_selection import chi2, SelectKBest, SelectPercentile, f_classif
import numpy as np
from sklearn.metrics import matthews_corrcoef

from skbio.stats.ordination import rda
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
## for word embedding
import gensim
import gensim.downloader as gensim_api

SVCWPrecision=[]
RFWPrecision=[]
LRWPrecision=[]
GBWPrecision=[]
DTWPrecision=[]
KNNWPrecision=[]

SVCPrecision=[]
RFPrecision=[]
LRPrecision=[]
GBPrecision=[]
DTPrecision=[]
KNNPrecision=[]

SVCWRecall=[]
RFWRecall=[]
LRWRecall=[]
GBWRecall=[]
DTWRecall=[]
KNNWRecall=[]


SVCRecall=[]
RFRecall=[]
LRRecall=[]
GBRecall=[]
DTRecall=[]
KNNRecall=[]


SVCWFscore=[]
RFWFscore=[]
LRWFscore=[]
GBWFscore=[]
DTWFscore=[]
KNNWFscore=[]


SVCFscore=[]
RFFscore=[]
LRFscore=[]
GBFscore=[]
DTFscore=[]
KNNFscore=[]

SVCWAUC=[]
RFWAUC=[]
LRWAUC=[]
GBWAUC=[]
DTWAUC=[]
KNNWAUC=[]

SVCAUC=[]
RFAUC=[]
LRAUC=[]
GBAUC=[]
DTAUC=[]
KNNAUC=[]


SVCMCC=[]
RFMCC=[]
LRMCC=[]
GBMCC=[]
DTMCC=[]
KNNMCC=[]


SVCError=[]
RFError=[]
LRError=[]
GBError=[]
DTError=[]
KNNError=[]


SVCTest=[]
RFTest=[]
LRTest=[]
GBTest=[]
DTTest=[]
KNNTest=[]

SVCTrain=[]
RFTrain=[]
LRTrain=[]
GBTrain=[]
DTTrain=[]
KNNTrain=[]

Environment_AccessP=[]
ExternalP=[]
Infrastructure_CreationP=[]
Infrastructure_SetupP=[]
Infrastructure_TemplatesP=[]
Infrastructure_VariablesP=[]
ReaderP=[]
Service_OperationsP=[]
WriterP=[]


Environment_AccessR=[]
ExternalR=[]
Infrastructure_CreationR=[]
Infrastructure_SetupR=[]
Infrastructure_TemplatesR=[]
Infrastructure_VariablesR=[]
ReaderR=[]
Service_OperationsR=[]
WriterR=[]


Environment_AccessF=[]
ExternalF=[]
Infrastructure_CreationF=[]
Infrastructure_SetupF=[]
Infrastructure_TemplatesF=[]
Infrastructure_VariablesF=[]
ReaderF=[]
Service_OperationsF=[]
WriterF=[]
AUC =[]
Brier=[]
BrierMed=[]
AUCMed=[]
WAUC=[]
SAUC=[]
correlatedFeatures=[]
redundantFeatures=[]
R_correlatedFeature=[]
R_RedundantFeatures=[]

correct=[]
incorrect=[]
config=[]
nonconfig=[]
df = pd.read_csv(r'/content/Model1.csv',encoding = "ISO-8859-1", names = ['Category', 'Content'])
#print(df)
df.dropna(how='all', axis=1, inplace=True)
#print(df.head())
## rename columns
df = df.rename(columns={"Content":"SourceCode"})
df.dropna(subset = ["SourceCode"], inplace=True)

def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and characters and thenstrip)
    #text = BeautifulSoup(text, "html.parser").text
    text = re.sub(r'\|\|\|', r' ' , text)
    # remove all single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', r' ' , text)
    text = re.sub(r'[^\w\s+]', ' ', str(text).lower().strip())
    text = re.sub(r'[0-9\.]+', ' ', str(text).lower().strip())
    text = re.sub(r'[,\'"#@-_$%&()*+.!:;<=>?]+', ' ', str(text).lower().strip())
    #text = re.sub(r'!"$%&()*+./:;<=>?[\\]^', ' ', str(text).lower().strip())
    text = '  '.join([w for w in text.split() if len(w) > 3])
    text = re.sub(r'[^\w\s]', ' ', str(text).lower().strip())
    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]

    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    ## back to string from list
    text = " ".join(lst_text)
    return text
lst_stopwords = nltk.corpus.stopwords.words("english")
lst_stopwords
df["SourceCode"] = df["SourceCode"].apply(lambda x:
                                          utils_preprocess_text(x, flg_stemm=True, flg_lemm=True,
                                                                lst_stopwords=lst_stopwords))


## WE SHUFFLE THE DATA
from sklearn.utils import shuffle
df = shuffle(df)
df.reset_index(inplace=True, drop=True)
#print(df.head())
from sklearn.utils import resample, check_random_state
DataList = df.values.tolist()
Categories = df['Category'].values.tolist()
print(Categories)
n_size = int(len(df) * 0.50)
## split dataset into bootstrap samples
def bootstrap_samples(data, n=100):
    samples = []
    for i in range(n):
        df_data = pd.DataFrame(data, columns=['Category', 'SourceCode'])
        sample = df_data.sample(n=len(df_data), replace=True)
        samples.append(sample)
    return samples

DataList = bootstrap_samples(DataList1, n=100, )
for i in range(100):
    print("This is iteration n°: ", i)
     print("This is iteration n°: ", i)
    #print(DataList[i])
    train = resample(DataList[i], replace=True, stratify=Categories,n_samples=df.shape[0])

    test = DataList[i][~DataList[i]['SourceCode'].isin(train['SourceCode'])]
    df_test = pd.DataFrame(test, columns=['Category', 'SourceCode'])
    df_train = pd.DataFrame(train, columns=['Category', 'SourceCode'])
    y_train = df_train["Category"].values
    y_test = df_test["Category"].values
    x_train = df_train["SourceCode"].values
    X_test = df_test["SourceCode"]
    print(test)

    ##DISPLAY ALL DATA

    pd.set_option('max_rows', 99999)
    pd.set_option('max_colwidth', 10000)
    pd.set_option('display.max_columns', 500)
    pd.describe_option('max_colwidth')
    #print(df_test)

## Tf-Idf (advanced variant of BoW)
    #create features by extracting information from the data
    #vectorizer = feature_extraction.text.TfidfVectorizer(max_features=1000, ngram_range=(1,2))
    #extrcat vocabulary and create feature matrix
    corpus = df_train["SourceCode"]
    #vectorizer.fit(corpus)
    #X_train = vectorizer.transform(corpus)
    #dic_vocabulary = vectorizer.vocabulary_

    from sklearn.feature_extraction.text import TfidfVectorizer


    vectorizer = feature_extraction.text.TfidfVectorizer(max_features=100, ngram_range=(1,1),stop_words='english')
    vectorizer.fit_transform(x_train) 

    

    X_train1 = vectorizer.fit_transform(corpus)
    X_test1 = vectorizer.transform(X_test)
    

    Features= SelectKBest(score_func=chi2, k=50) #, k='all' )
    #

    #Features = SelectPercentile (f_classif, percentile=10)

    
    Features.fit(X_train1,y_train)






   






    

    #####SELECT FEATURES NAMES AND CHI2 SCORES

    #print(Features.scores_)
    X_trainF=Features.transform(X_train1)

    #print(X_trainF)
    ##generate a Document Term Matrix of features and scores in all documents

    mask = Features.get_support() 
    #print(mask)
    
        ## Filtered features using chi-square
    new_features = [] # The list of your K best features
    for bool, feature in zip(mask, vectorizer.get_feature_names()):
      if bool:
        new_features.append(feature)
        
    DTM = pd.DataFrame(X_trainF.todense(), columns=new_features)
    #DTM = pd.DataFrame(X_trainF.toarray(), columns=new_features)
    #DTM = DTM[(DTM.T != 0).any()]
    #print(DTM)
    print(new_features)
    import seaborn as sns

    #sns.regplot(x=DTM['apach'], y=DTM['ansibl'])   

    ##COMPUTE CORRELATION 

    DTM1=DTM.corr(method='spearman')
    cf=0
    #print(DTM1)
    #print(DTM.iat[0,2])
    List=[]
    FEATUESNAMES=[]
    FEATURESINDEX=[]
    FEATURESCORRELATION=[]
    for i in range(49):
      for j in range(49):
        if i!=j and DTM1.iat[i,j]>=0.7:
          #print(DTM.iat[i,j])
          #print(i,j)
          colname = DTM1.columns[j]
          rowname = DTM1.columns[i]
          #print(colname, rowname)
          FEATUESNAMES.append(colname)
          R_correlatedFeature.append(colname)
          FEATURESINDEX.append(j)
          FEATURESCORRELATION.append(DTM1.iat[i,j])
          
        
          List.append(DTM1.iat[i,j])
    ## KEEP ONE OCCURRENCE
    TOBEREMOVEDFEATURES = list(dict.fromkeys(FEATURESINDEX))
    #print(TOBEREMOVEDFEATURES)
    #print(FEATUESNAMES)
    #print(FEATURESINDEX)
    #print(FEATURESCORRELATION)    

    #REMOVE CORRELATED FEATURES
    for i in range(len(TOBEREMOVEDFEATURES)):
     #print(i)
     DTM.drop(DTM.columns[i], axis=1, inplace=True)

    #print(DTM)
    #print(TOBEREMOVEDFEATURES)



    Nb_correlated_features=len(TOBEREMOVEDFEATURES)
    #print(Nb_correlated_features)
    correlatedFeatures.append(Nb_correlated_features)
    #print(X_trainF.getrow(5))

    #for i in range(len(TOBEREMOVEDFEATURES)):
     #X_trainF.delete(i,OBEREMOVEDFEATURES[i] )

    #print(X_trainF)


    #X_trainFN = Features.fit_transform(DTM,y_train) 
    #print(X_trainFN)

    DTM.to_csv('/content/Features-After-correlation.csv')
    from rpy2.robjects import r, pandas2ri
    pandas2ri.activate()
    import rpy2.robjects as ro


    ##USING REDUN R PACKAGE
    # Defining the R script and loading the instance in Python
    robjects.r.source("/content/Redundancy.R", encoding="utf-8")
    # Loading the function we have defined in R.
    redundant_function_r = robjects.globalenv['redundant']
    df = pd.read_csv("/content/Features-After-correlation.csv")
    #df_r = pandas2ri.ri2py(df)
    df_r = ro.conversion.py2rpy(df)
    
    df_result_r = redundant_function_r(df_r)
    #print(df_result_r)
    
       Redundant =ro.conversion.rpy2py(df_result_r)
       #print(Redundant)
    #print("############")
    #print(Redundant[3])
    
    Output=Redundant[3]
    Output = ' '.join(Redundant[3])
    #print(Output)
    print("############")
    Features_List=Redundant[3]
    #print(Features_List)
    if len(Output) == 0 or Output.isspace():
      print("No Redundant Features")
    else :   
      ListFeaturesRedundant = list(Output.split(" "))
      #print(ListFeaturesRedundant)
    
    if "character(0)" in Features_List:
      print("yes") 
    else:
      #print(Features_List)  
      for i in Features_List:
        R_RedundantFeatures.append(i)
        redundantFeatures.append(len(Features_List))
        #print(i)
        column=str(i)
        #print(column)
        index_no = DTM.columns.get_loc(column)
        DTM.drop(DTM.columns[index_no], axis=1, inplace=True)  
    
    #print(DTM)
    result = DTM.mean()
    #print(result) 
    columns=[]
    medians=[]
    





    #drop features names from DTM
    #X_train
    sparse_DTM = DTM.astype(pd.SparseDtype("float64",0))
    X_TRAIN_AFTER_RED = sparse_DTM.sparse.to_coo().tocsr()
    #print(X_TRAIN_AFTER_RED)

    #X_trainF=Features.fit_transform(X_TRAIN_AFTER_RED,y_train)
    #print("new XTRAIN")
    #print(X_trainF)
    
    n = DTM.shape[1]
    #print(n)
    
       #################
    ##WORKING SOLUTION
    feature_names = vectorizer.get_feature_names() 
    feature_names = [feature_names[i] for i in Features.get_support(indices=True)]
    #print(Features.scores_)
    #print(Features.get_support(indices=True))
    #print(feature_names)
    dataframeScores = pd.DataFrame(Features.scores_)
    #print(dataframeScores)

    dataframeNames = pd.DataFrame(feature_names)
    #print(dataframeNames)
    #print(Features.get_params)
    #continue adding the 2 datframes to a dict, and then start removing features
    ###############





   
    #print(dataframe)
    #print(X_trainF.scores_)
    #print(X_trainF.shape)
    X_testF=Features.transform(X_test1)

        #classifier =svm.SVC(probability=True, C=1000, max_iter=-1, kernel="rbf", gamma= 'scale')
    #classifier= LogisticRegression(random_state=42, C=50, penalty='l2', solver='newton-cg', max_iter=1000)
    classifier = RandomForestClassifier(n_estimators=200, max_depth=20,random_state=42, class_weight="balanced")
    #classifier = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0,  max_depth=10, random_state=42)
    #classifier = KNeighborsClassifier(n_neighbors=10, metric= 'euclidean', weights= 'distance')

     classifier.fit(X_TRAIN_AFTER_RED, y_train)
    #classifier.fit(X_train, y_train)


  
    types=np.unique(y_train)
    display(eli5.show_weights(classifier ,vec=vectorizer,top=20, 
                  #target_names=types))
    

       #for i in range(Nb_correlated_features):
       #X_TEST=remove_cols()
    red= len(Features_List)
    NB=Nb_correlated_features+red
    X_TEST = csr_matrix(np.delete(X_testF.toarray(),np.s_[1:NB+1], axis=1))
    #X_TEST = csr_matrix(np.delete(X_test1.toarray(),np.s_[1:NB+1], axis=1))
   
    #print(X_TEST.shape)
    #print(X_TEST)
    predicted = classifier.predict(X_TEST)
    #print(predicted)
    predicted_prob = classifier.predict_proba(X_TEST)
    List_predicted_prob= predicted_prob.tolist()
    
    flat_List_predicted_prob = [item for sublist in List_predicted_prob for item in sublist]
    #print(flat_List_predicted_prob)


    
    #to show which features are importance for generalization, we compute importance on test set
    #perm = PermutationImportance(classifier).fit(X_testF.toarray(), y_test)
    perm = PermutationImportance(classifier).fit(X_TEST.toarray(), y_test)

    types=np.unique(y_train)
    display(eli5.show_weights(perm ,vec=vectorizer,top=20, 
     #             target_names=types.tolist()))
    

    
    classes = np.unique(y_test)
    
    y_test_array = pd.get_dummies(y_test, drop_first=False).values
    
    #List_y_test_array=y_test_array.tolist()
    
    
    auc = metrics.roc_auc_score(y_test,predicted_prob[:, 1])
    for i in range(len(classes)):
     brier= brier_score_loss(y_test_array[:, i],predicted_prob[:, i])
     print(classes[i], '--brier--->', brier)
     Brier.append(brier)
    #Brier.append("###") 
    Brier.sort()
    Med = len(Brier) // 2
    BrierMed.append((Brier[Med] + Brier[~Med]) / 2)
    Brier.clear()

    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:, i], predicted_prob[:, i])
        auroc = round(metrics.auc(fpr, tpr), 4)
        print('LRE', classes[i], '--AUC--->', auroc)
        AUC.append(auroc)
    AUC.append("###")


    
    
    print("AUC:", auc)
   
    SAUC.append(auc)
    print("Brier median")
    for i in range(len(BrierMed)):
        print(BrierMed[i])

    print("AUC median")
    for i in range(len(SAUC)):
        print(SAUC[i])

  
print("Brier")
for i in range (len(BrierMed)):
    print(BrierMed[i])  
print("AUC ")
for i in range(len(SAUC)):   
      print(SAUC[i])

