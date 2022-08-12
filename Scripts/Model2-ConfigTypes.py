# -*- coding: utf-8 -*-


## for data
## for interpretation

!pip install lime
import lime
from lime import lime_tabular


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

Creation=[]
External=[]
Operations=[]
Reader=[]
Declaration=[]
Templates=[]
Variable=[]
Access=[]
Setup=[]



df = pd.read_csv('/content/Model2.csv')
#print(df.head())
## rename columns
df = df.rename(columns={"Content":"SourceCode"})
df.dropna(subset = ["SourceCode"], inplace=True)
## print 5 random rows
df.rename({"Unnamed: 2": "a"}, axis="columns",inplace=True)
#drop the column
#df.drop(["a"], axis=1, inplace=True)
#print(df.sample(5))


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
n_size = int(len(df) * 0.60)
## split dataset
for i in range(100):
    print("This is iteration n°: ", i)
    # train = resample(DataList)


    #train = resample(DataList, replace=True, stratify=Categories,n_samples=df.shape[0])
    train = resample(DataList, replace=True, stratify=Categories,n_samples=df.shape[0])
    #train = resample(DataList, replace=True, stratify=Categories,n_samples=75)
    test = [item for item in DataList if item not in train]
    df_test = pd.DataFrame(test, columns = ['Category','SourceCode'])
    df_train = pd.DataFrame(train, columns = ['Category','SourceCode'])
    y_train = df_train["Category"].values
    y_test = df_test["Category"].values
    #print(y_test)
    x_train = df_train["SourceCode"].values
    X_test = df_test["SourceCode"]
    X_testOriginal = df_train["SourceCode"].values

    ##DISPLAY ALL DATA

    pd.set_option('max_rows', 99999)
    pd.set_option('max_colwidth', 10000)
    pd.set_option('display.max_columns', 500)
    pd.describe_option('max_colwidth')
    #print(df_test)

## Tf-Idf (advanced variant of BoW)
    #create features by extracting information from the data

    corpus = df_train["SourceCode"]
    
    from sklearn.feature_extraction.text import TfidfVectorizer


    vectorizer = feature_extraction.text.TfidfVectorizer(max_features=100, ngram_range=(1,1),stop_words='english')


    X_train1 = vectorizer.fit_transform(corpus)
    X_test1 = vectorizer.transform(X_test)
    Features= SelectKBest(score_func=chi2, k=100) #, k='all' )
    #


    import lime
    from lime import lime_text
    from lime.lime_text import LimeTextExplainer
    from sklearn.pipeline import make_pipeline
    class_names =np.unique(y_test)

    explainer = LimeTextExplainer(class_names=class_names) 

    model =make_pipeline(vectorizer,Features, classifier)
    model.fit(x_train, y_train)
    idx = 5
    exp = explainer.explain_instance(X_test[idx], 
                                 model.predict_proba, 
                                 num_features=10)
    print('Document id: %d' % idx)
    display(exp.show_in_notebook(text=True))
    display(exp.show_in_notebook(text=False))
    print('R2 score: {:.3f}'.format(exp.score))
    exp.save_to_file('/content/Lime.html')
    
    #print(Features)
    Features.fit(X_train1,y_train)
    

    #####SELECT FEATURES NAMES AND CHI2 SCORES

    print(Features.scores_)
    X_trainF=Features.transform(X_train1)

    #print(X_trainF.shape)
    ##generate a Document Term Matrix of features and scores in all documents

    mask = Features.get_support() 
    ## Filtered features using chi-square
    new_features = [] # The list of your K best features
    for bool, feature in zip(mask, vectorizer.get_feature_names()):
      if bool:
        new_features.append(feature)
        
    DTM = pd.DataFrame(X_trainF.todense(), columns=new_features)
    
    import seaborn as sns

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
    #REMOVE CORRELATED FEATURES
    for i in range(len(TOBEREMOVEDFEATURES)):
     #print(i)
     DTM.drop(DTM.columns[i], axis=1, inplace=True)

    Nb_correlated_features=len(TOBEREMOVEDFEATURES)
    #print(Nb_correlated_features)
    correlatedFeatures.append(Nb_correlated_features)
    
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
    #df_r =ro.conversion.rpy2py(df)
    #df_r =com.convert_to_r_dataframe(df)
    df_result_r = redundant_function_r(df_r)
    #print(df_result_r)
    
    #pd_df = pandas2ri.ri2py_dataframe(df_result_r)
    #print(pd_df)
    Redundant =ro.conversion.rpy2py(df_result_r)
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
    
    #drop features nemes from DTM
    #X_train
    sparse_DTM = DTM.astype(pd.SparseDtype("float64",0))
    X_TRAIN_AFTER_RED = sparse_DTM.sparse.to_coo().tocsr()
     ##REDUNDANCY

    def optimise_cca_cv(X, y, n_comp):
       # Define PLS object
       cca = CCA(n_components=n_comp, scale=True)
       # Cross-validation
       y_cv = cross_val_predict(cca, X, y, cv=10)
       # Calculate scores
       r2 = r2_score(y, y_cv)
       mse = mean_squared_error(y, y_cv)
       rpd = y.std()/np.sqrt(mse)
       return (y_cv, r2, mse, rpd)

    



    def do_analysis(DTM,FeatureToBeInvestigated,RemainingFeatures, n_components=-1):

      Y_cols = FeatureToBeInvestigated
      X_cols = RemainingFeatures
      Y = DTM[Y_cols]
      X = DTM[X_cols]  
      if n_components == -1:
        r2s = []
        mses = []
        rpds = []
        xticks = np.arange(1, X.shape[1] + 1)
        for n_comp in xticks:
          y_cv, r2, mse, rpd = optimise_cca_cv(X, Y, n_comp)
          r2s.append(r2)
          mses.append(mse)
          rpds.append(rpd)

        n_components = np.argmin(mses) + 1
      cca = CCA(n_components=n_components, scale=True)
      cca.fit(X, Y)
      loadings = pd.DataFrame(cca.x_loadings_)
      scores = pd.DataFrame(cca.x_scores_)
      X_rows_dict = {i : X_cols[i] for i in range(0, len(X_cols))}
      X_cols_dict = {i : 'LV' + str(i+1) for i in range(0, n_components)}
      loadings.rename(index=X_rows_dict, columns=X_cols_dict, inplace=True)
      #print(loadings) 
      #print(scores)    
      rda_res = rda(Y, X, scale_Y=True)
      return (loadings)

    
    n = DTM.shape[1]
    
    

    

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
    

    X_testF=Features.transform(X_test1)

    
    from imblearn.over_sampling import SMOTE
    X_train, y_train = SMOTE(k_neighbors=2).fit_resample(X_TRAIN_AFTER_RED, y_train)
    #X_train, y_train = SMOTE().fit_resample(X_trainF, y_train)
    
    
      #classifier =svm.SVC(probability=True, C=1000, max_iter=-1, kernel="rbf", gamma= 'scale', decision_function_shape='ovr')
    #classifier= LogisticRegression(random_state=42, C=50, penalty='l2', solver='newton-cg', max_iter=1000, multi_class="ovr")
    classifier = RandomForestClassifier(n_estimators=200, max_depth=20,random_state=42, class_weight="balanced")
    #classifier = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0,  max_depth=10, random_state=42)
    #classifier = KNeighborsClassifier(n_neighbors=10, metric= 'euclidean', weights= 'distance')

# train classifier
    #model["classifier"].fit(X_train, y_train)
    classifier.fit(X_train, y_train)
    #clf.fit(X_train, y_train)
    #perm = PermutationImportance(classifier).fit(X_testF, y_test)

    types=np.unique(y_train)


    import lime
    from lime import lime_text
    from lime.lime_text import LimeTextExplainer
    from sklearn.pipeline import make_pipeline
    class_names =np.unique(y_test)

    explainer = LimeTextExplainer(class_names=class_names) 

    model =make_pipeline(vectorizer,Features, classifier)
    model.fit(X_train, y_train)
    idx = 5
    exp = explainer.explain_instance(X_test[idx], 
                                 model.predict_proba, 
                                 num_features=10)
    print('Document id: %d' % idx)
    print('Probability (Not-Configuration) = {:.3f}'.format(model.predict_proba([X_test[idx]])[0,1]))
    print("true class:" % class_names[y_test[idx]])
    display(exp.show_in_notebook(text=True))

    print('R2 score: {:.3f}'.format(exp.score))
    exp.save_to_file('/tmp/Lime.html')
    #print('True class: %s' % class_names1[y_test[idx]])
    #exp.show_in_notebook(text=True)



   #for i in range(Nb_correlated_features):
       #X_TEST=remove_cols()
    red= len(Features_List)
    NB=Nb_correlated_features+red
    X_TEST = csr_matrix(np.delete(X_testF.toarray(),np.s_[1:NB+1], axis=1))
    #X_TEST = csr_matrix(np.delete(X_test1.toarray(),np.s_[1:NB+1], axis=1))
   
    #print(X_TEST.shape)
    predicted = classifier.predict(X_TEST)
    predicted_prob = classifier.predict_proba(X_TEST)
    

    List_predicted_prob= predicted_prob.tolist()
    
    flat_List_predicted_prob = [item for sublist in List_predicted_prob for item in sublist]
    #print(flat_List_predicted_prob)


    
    #to show which features are importance for generalization, we compute importance on test set
    #perm = PermutationImportance(classifier).fit(X_testF.toarray(), y_test)
    perm = PermutationImportance(classifier).fit(X_TEST.toarray(), y_test)

    types=np.unique(y_train)
    display(eli5.show_weights(perm ,vec=vectorizer,top=10, 
                  target_names=types.tolist()))
    #display(eli5.show_prediction(perm, X_testF[0], vec=vectorizer, 
                #     target_names=np.unique(y_test), show_feature_values=True))

    classes = np.unique(y_test)
    print(classes)
    #print(np.unique(predicted))
    TestAccuracy = metrics.accuracy_score(y_test, predicted)
    MCC= matthews_corrcoef(y_test, predicted)
    Precision=metrics.precision_score(y_test, predicted, average=None)
    WPrecision=metrics.precision_score(y_test, predicted, average= 'weighted')
    Recall=metrics.recall_score(y_test, predicted, average=None)
    WRecall=metrics.recall_score(y_test, predicted, average= 'weighted')
    F1score=metrics.f1_score(y_test, predicted, average=None)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values
    #print(y_test)
    List_y_test_array=y_test_array.tolist()
    
    flat_List_y_test_array = [item for sublist in List_y_test_array for item in sublist]
    #print(flat_List_y_test_array)

    List_score=[]
    #for sublist in t:
    #for item in sublist:
     #   flat_list.append(item)
    
    for i in range(0,len(flat_List_y_test_array)):
      score=0
      score=flat_List_predicted_prob[i]-flat_List_y_test_array[i]
      if score<0:
       List_score.append(abs(score))

  
    WF1score=metrics.f1_score(y_test, predicted, average= 'weighted')
    auc = metrics.roc_auc_score(y_test,predicted_prob, multi_class="ovr", average= 'macro')
    Wauc = metrics.roc_auc_score(y_test,predicted_prob, multi_class="ovo", average= 'weighted')
    for i in range(len(classes)):
     brier= brier_score_loss(y_test_array[:, i],predicted_prob[:, i])
     #print(classes[i], '--brier--->', brier)
     Brier.append(brier)
    #Brier.append("###") 
    Brier.sort()
    Med = len(Brier) // 2
    BrierMed.append((Brier[Med] + Brier[~Med]) / 2)
    #Brier.clear()
    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:, i], predicted_prob[:, i])
        auroc = round(metrics.auc(fpr, tpr), 4)
        print('LRE', classes[i], '--AUC--->', auroc)
        AUC.append(auroc)
   #print(predicted.shape, y_test.shape)
    incorrect1 = np.where(predicted!=y_test)[0]
    #print(incorrect1)
    correct1 = np.where(predicted==y_test)[0]

    #print ("Found %d incorrect labels" % len(incorrect1))
    #print ("Found %d correct labels" % len(correct1))

    #correct.append(len(correct1))
    #incorrect.append(len(incorrect1))
    external=0
    reader=0
    declaration=0
    operations=0
    access=0
    creation=0
    templates=0
    variables=0
    setup=0
    for incorrect in range(0, len( predicted)):
      #print("Predicted:::"+predicted[incorrect] )
      brier=brier_score_loss(y_test_array[incorrect],predicted_prob[incorrect])
      print("AUC:", auc)
    print("Wauc:", Wauc)
    WAUC.append(Wauc)
    SAUC.append(auc)
    print("WAUC ")
    for i in range(len(WAUC)):   
      print(WAUC[i])   
    print("AUC")
    for i in range (len(SAUC)):
      print(SAUC[i])  
    #print("Brier")
    #for i in range (len(Brier)):
     # print(Brier[i])    
    print("Brier median")
    for i in range(len(BrierMed)):
        print(BrierMed[i])

  
print("BrierMed")
for i in range (len(BrierMed)):
    print(BrierMed[i])  
print("SAUC ")
for i in range(len(SAUC)):   
      print(SAUC[i]) 

print("WAUC ")
for i in range(len(WAUC)):   
      print(WAUC[i])   


