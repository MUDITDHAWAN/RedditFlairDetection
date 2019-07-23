
import nltk

import pandas as pd

### reading the csv file into a Data frame
df = pd.read_csv("datafinal.csv")
###AskIndia              4494                                                                                              Business/Finance      4494                                                                                              Food                  1151                                                                                              Non-Political         4494                                                                                              ###Photography           2549                                                                                              Policy/Economy        3256                                                                                              Politics              4494                                                                                              Scheduled              798                                                                                              Science/###Technology    4494                                                                                              [R]eddiquette         2767
###
###
###
### saving different columns a separate dataframes to be processed differently
### as per the colum heading we saved them in the csv file
saved_column = df.entire_text
flair = df.flair
author=df.author
score=df.score
timestamp=df.timestamp
numComms=df.numComms
url=df.url
title=df.title
###concating the data back into one single data frame (testing purpose)
X=pd.concat([author, score, title, url, numComms,timestamp],axis=1)

print(X)
print(len(X))
### Figuring out the vriation of data over different flairs
print(df.groupby('flair').size())
###flair
###AskIndia              14561
###Business/Finance       7117
###Food                   1151
###Non-Political         25084
###Photography            2549
###Policy/Economy         3256
###Politics              22495
###Scheduled               798
###Science/Technology     4494
###[R]eddiquette          2767
######dtype: int64
###

### finding out that the data is largely skewed I read about ways to solve this

###                1                  ###
### large data skewness by making the dataset smaller by leaving out some of the
### data from the classes which have excess of data

### variables to count the number of examples counted from each class which has
### excess
count_Ask=0
count_Pol=0
count_NonPol=0
count_BF=0


### lists to store the data of the new training data
training_X_with_all_the_features_extracted=list()
Y_flairs=list()

###this is done because of ValueError: array is too big;
### `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
### Explained later
training_X_for_storing_the_text_columns_as_a_single_string=list()

### parsing through the dataframe by indices-- rows ( pd.['<column_header>'][index]
### used ) for finding the corresponding values and uploading them tuples
### checking for each row the corresponding flair and correspondly deciding
### whether to include it in new data or not depending upon the previously
### count of that flair
### 4494 decided on the basis of Science and Technology data and the
### other classes which have data close to it
for ind in X.index:

    ### for AskIndia class
    if(flair[ind]=="AskIndia"):
        ### 4494 decided on the basis of Science and Technology data and the
        ### other classes which have data close to it
        if(count_Ask<4494):
            ### appending data as tuples into the lists so that they can be
            ### converted into Data frames after this
            training_X_with_all_the_features_extracted.append((X["author"][ind], X["score"][ind], X["title"][ind], X["url"][ind], X["numComms"][ind],X["timestamp"][ind]))
            count_Ask+=1
            training_X_for_storing_the_text_columns_as_a_single_string.append(str(X["author"][ind]+" " +X["title"][ind]+" " +X["url"][ind]))
            Y_flairs.append(flair[ind])

    ### for Business/Finance class
    elif(flair[ind]=="Business/Finance"):

        if(count_BF<4494):

            training_X_with_all_the_features_extracted.append((X["author"][ind], X["score"][ind], X["title"][ind], X["url"][ind], X["numComms"][ind],X["timestamp"][ind]))
            count_BF+=1
            training_X_for_storing_the_text_columns_as_a_single_string.append(str(X["author"][ind]+" " +X["title"][ind]+" " +X["url"][ind]))
            Y_flairs.append(flair[ind])

    ### for Non-Political class
    elif(flair[ind]=="Non-Political"):
        if(count_NonPol<4494):
            training_X_with_all_the_features_extracted.append((X["author"][ind], X["score"][ind], X["title"][ind], X["url"][ind], X["numComms"][ind],X["timestamp"][ind]))
            count_NonPol+=1
            training_X_for_storing_the_text_columns_as_a_single_string.append(str(X["author"][ind]+" " +X["title"][ind]+" " +X["url"][ind]))
            Y_flairs.append(flair[ind])

    ### for Politics class
    elif(flair[ind]=="Politics"):
        if(count_Pol<4494):
            training_X_with_all_the_features_extracted.append((X["author"][ind], X["score"][ind], X["title"][ind], X["url"][ind], X["numComms"][ind],X["timestamp"][ind]))
            count_Pol+=1
            training_X_for_storing_the_text_columns_as_a_single_string.append(str(X["author"][ind]+" " +X["title"][ind]+" " +X["url"][ind]))
            Y_flairs.append(flair[ind])

    ### other remaining classes have less data, therefore no restricitons on them
    else:
        training_X_with_all_the_features_extracted.append((X["author"][ind], X["score"][ind], X["title"][ind], X["url"][ind], X["numComms"][ind],X["timestamp"][ind]))
        Y_flairs.append(flair[ind])
        training_X_for_storing_the_text_columns_as_a_single_string.append(str(X["author"][ind]+" " +X["title"][ind]+" " +X["url"][ind]))

### converting the lsits of tuples into corresonding data frames
X = pd.DataFrame(training_X_with_all_the_features_extracted)
Y = pd.DataFrame(Y_flairs)

X_other= pd.DataFrame(training_X_for_storing_the_text_columns_as_a_single_string)
# print(X')
### now using the new data, I tried to use all the columns for training the model
### for better results but after individually vectorizing each column with text
### data sepately using TfidfVectorizer() wich would return a csr_matrix
### to join them with the remaining numericall columns i had to convert it to a
### dense matrix using cs_matrix.todense() but was unable to this because
###ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than
### the maximum possible size. which I believe is because the low  computional
### power of my laptop

x_author=X[0]
x_score=X[1]
x_title=X[2]
x_url=X[3]
x_numComms=X[4]
x_timestamp=X[5]


print(Y.groupby(0).size())
### AskIndia              4494
### Business/Finance      4494
### Food                  1151
### Non-Political         4494
### Photography           2549
### Policy/Economy        3256
### Politics              4494                                                                                              ###Scheduled              798                                                                                              Science/###Technology    4494                                                                                              [R]eddiquette         2767


### importing TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

###########          2              ####################

# vectorizer_author = TfidfVectorizer()
# vectorizer_title = TfidfVectorizer()
# vectorizer_url = TfidfVectorizer()
# x_author = vectorizer_author.fit_transform(x_author).todense()
# x_title = vectorizer_title.fit_transform(x_title).todense()
# x_url = vectorizer_url.fit_transform(x_url).todense()
#
# x_author=pd.DataFrame(x_author,columns=vectorizer_author.get_feature_names())
# X_Data=pd.concat([x_score, x_numComms,x_timestamp,x_author,x_title,x_url],axis=1)
#
### in the above statements i got the ValueError and therefore had to look for
### some other method to use the data




###            3 (a)    ####
### used ony the title of the the posts to predict with the new data
train_vectors = vectorizer.fit_transform(x_title)



###imported preprocessing for encoding the flairs to be fed into the model
from sklearn import preprocessing
#
### label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
#
### encoded flairs to be used for as Y_data
encodedflair= label_encoder.fit_transform(Y)

### train_data: test_data= 70:30
x_train=train_vectors[:23094]
y_train=encodedflair[:23094]
x_test=train_vectors[23094:]
y_test=encodedflair[23094:]

###    4    ###
### tried using SMOTE from the imblearn package for synthetic development of
### data for increasing the number for the classes with less data
### couldn't complete due to lack of knowledge and time

# from imblearn.over_sampling import SMOTE
# smote= SMOTE('minority')
#
# xtrain, ytrain= smote.fit_sample(train_vectors[:23094], encodedflair[:23094])
#
# print(xtrain.shape[0])
###

###     5   ####
### uing ensemble learning- dividing the data into parts such that the classes
### having larger data are divided into goups randomly with classes with fewer
### data points and then ml model is run on all the groups and at the end the
### prediction is based on the max of the predictions of all the models
### couldn't complete again due to lack of knowledge and time

### imported different metric system to evaluate performance of the model
from  sklearn.metrics  import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score


#### training data with Naive Bayes model usinf sklearn's MultinomialNB()
from sklearn.naive_bayes import MultinomialNB
clf_MN_NB_a = MultinomialNB().fit(x_train, y_train)

predicted = clf_MN_NB_a.predict(x_test)
print(accuracy_score(y_test,predicted))

precision, recall, fscore, support = score(encodedflair[23094:], predicted)
### table printed column for each class
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

### output :
###0.157118318682429
###C:\Users\MUDIT DHAWAN\AppData\Local\Programs\Python\Python37-32\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
###  'precision', 'predicted', average, warn_for)
###C:\Users\MUDIT DHAWAN\AppData\Local\Programs\Python\Python37-32\lib\site-packages\sklearn\metrics\classification.py:1439: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
###  'recall', 'true', average, warn_for)
### precision: [0.         0.61456753 1.         0.         0.88636364 0.85714286
###           0.         1.         0.80285036 0.        ]
### recall: [0.         0.62307692 0.01474926 0.         0.0248566  0.01109057
###          0.         0.68136273 0.14568966 0.        ]
### fscore: [0.         0.61879297 0.02906977 0.         0.0483571  0.02189781
###        0.         0.81048868 0.24662532 0.        ]
### support: [   0 1300  678    0 1569 1623    0  499 2320 1908]

### training data with Neural Network usinf sklearn's  MLPClassifier()
from sklearn.neural_network import MLPClassifier

clf_NN_a = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

clf_NN_a.fit(x_train, y_train)
predicted=clf_NN_a.predict(x_test)

print(accuracy_score(y_test,predicted))

precision, recall, fscore, support = score(y_test, predicted)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

### output:

###0.35445084369000707
### C:\Users\MUDIT DHAWAN\AppData\Local\Programs\Python\Python37-32\lib\site-packages\sklearn\metrics\classification.py:1439: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
### 'recall', 'true', average, warn_for)
### precision: [0.         0.51559516 0.83152174 0.         0.77163462 0.71770972
###           0.         0.94936709 0.70604148 0.5464191 ]
### recall: [0.         0.62307692 0.22566372 0.         0.40917782 0.33210105
###     0.         0.75150301 0.3375     0.10796646]
###   fscore: [0.         0.56426332 0.3549884  0.         0.53477718 0.45408593
### 0.         0.83892617 0.45669291 0.18030635]
### support: [   0 1300  678    0 1569 1623    0  499 2320 1908]

###############          3(b)       #############
### using title of the post, url and the author if the post to predict the
### flair, I am able to use this withe converting the the csr_matrix into the
### matrix by joing the three columnes earlier only in a single string and the
### using the vectorizer as a sing entity
train_vectors = vectorizer.fit_transform(X_other[0])

### train_data: test_data= 70:30
x_train=train_vectors[:23094]
y_train=encodedflair[:23094]
x_test=train_vectors[23094:]
y_test=encodedflair[23094:]

### imported different metric system to evaluate performance of the model
from  sklearn.metrics  import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score



clf_MN_NB_b = MultinomialNB().fit(x_train, y_train)

predicted = clf_MN_NB_b.predict(x_test)
print(accuracy_score(y_test,predicted))

precision, recall, fscore, support = score(encodedflair[23094:], predicted)
### table printed column for each class
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

######## Output:
###0.10043447509346266
###C:\Users\MUDIT DHAWAN\AppData\Local\Programs\Python\Python37-32\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
###  'precision', 'predicted', average, warn_for)
###C:\Users\MUDIT DHAWAN\AppData\Local\Programs\Python\Python37-32\lib\site-packages\sklearn\metrics\classification.py:1439: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
###  'recall', 'true', average, warn_for)
###precision: [0.         0.68126747 0.         0.         0.5        0.75
### 0.         1.         0.9375     0.        ]
###recall: [0.         0.56230769 0.         0.         0.00063735 0.00369686
### 0.         0.00200401 0.10991379 0.        ]
###fscore: [0.         0.61609777 0.         0.         0.00127307 0.00735745
### 0.         0.004      0.19675926 0.        ]
###support: [   0 1300  678    0 1569 1623    0  499 2320 1908]

### using Neural network
clf_NN_b = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

clf_NN_b.fit(x_train, y_train)
predicted=clf_NN_b.predict(x_test)

print(accuracy_score(y_test,predicted))

precision, recall, fscore, support = score(y_test, predicted)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

#### Output:
###0.3975952308780438
###C:\Users\MUDIT DHAWAN\AppData\Local\Programs\Python\Python37-32\lib\site-packages\sklearn\metrics\classification.py:1145: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
###  'recall', 'true', average, warn_for)
###precision: [0.         0.59243086 0.82608696 0.         0.87751938 0.66781609
### 0.         0.76876268 0.65662651 0.67088608]
###recall: [0.         0.62615385 0.28023599 0.         0.36073932 0.35797905
### 0.         0.75951904 0.42284483 0.22222222]
###fscore: [0.         0.60882573 0.4185022  0.         0.51129178 0.46610509
### 0.         0.7641129  0.51442056 0.33385827]
###support: [   0 1300  678    0 1569 1623    0  499 2320 1908]
