import numpy as np
import re
import string
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim import utils
from nltk.corpus import stopwords

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
import os

import warnings
warnings.filterwarnings("ignore")


def textClean(text):
    """
    Get rid of the non-letter and non-number characters
    """
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return (text)


def cleanup(text):
    text = textClean(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def constructLabeledSentences(data):
    sentences = []
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
        #if index == 1:
         #   print(row, sentences)
    return sentences


def preProcessing(path,vector_dimension=300):
    """
    Generate Doc2Vec training and testing data
    """
    data = pd.read_csv(path)

    missing_rows = []
    for i in range(len(data)):
        if data.loc[i, 'text'] != data.loc[i, 'text']:
            missing_rows.append(i)
    data = data.drop(missing_rows).reset_index().drop(['index','id'],axis=1)

    for i in range(len(data)):
        data.loc[i, 'text'] = cleanup(data.loc[i,'text'])

    x = constructLabeledSentences(data['text'])
    y = data['label'].values
    #print(y)

    text_model = Doc2Vec(min_count=1, window=5, vector_size=vector_dimension, sample=1e-4, negative=5, workers=7, epochs=10,
                         seed=1)
    text_model.build_vocab(x)
    text_model.train(x, total_examples=text_model.corpus_count, epochs=text_model.iter)

    train_size = int(0.8 * len(x))
    test_size = len(x) - train_size

    text_train_arrays = np.zeros((train_size, vector_dimension))
    text_test_arrays = np.zeros((test_size, vector_dimension))
    train_labels = np.zeros(train_size)
    test_labels = np.zeros(test_size)

    for i in range(train_size):
        text_train_arrays[i] = text_model.docvecs['Text_' + str(i)]
        train_labels[i] = y[i]

    j = 0
    for i in range(train_size, train_size + test_size):
        text_test_arrays[j] = text_model.docvecs['Text_' + str(i)]
        test_labels[j] = y[i]
        j = j + 1

    return text_train_arrays, text_test_arrays, train_labels, test_labels



def plot_cmat(yte, ypred):
    skplt.plot_confusion_matrix(yte,ypred)
    plt.show()

xtr,xte,ytr,yte = preProcessing('./fake-news-dataset/train.csv')
#print(xtr)
print('\n\n\nxtr')
print(xtr)
print('\n\n\nxte')
print(xte)
print('\n\n\nytr')
print(ytr)
print('\n\n\nyte')
print(yte)


def accuracy(yte, y_pred, model):
    m = yte.shape[0]
    n = (yte != y_pred).sum()
    #print(m, yte)
    print("Accuracy of "+ model + " = " + format((m-n)/m*100, '.2f') + "%")

def model_classifier(xte, yte, xtr, ytr, model):
    if model == "nb":
        gnb = GaussianNB()
        gnb.fit(xtr,ytr)
        y_pred = gnb.predict(xte)
        accuracy(yte, y_pred, "Naive Bayes")
    elif model == "svm":
        clf = SVC()
        clf.fit(xtr, ytr)
        y_pred = clf.predict(xte)
        accuracy(yte, y_pred, "SVM")
    else:
        logreg = LogisticRegression()
        logreg.fit(xtr,ytr)
        y_pred = logreg.predict(xte)
        accuracy(yte, y_pred, "Logistic Regression")
        
    plot_cmat(yte, y_pred)
    
models = ["nb", "svm", "lr"]
for i in range(0,3):
    model_classifier(xte, yte, xtr, ytr, models[i])
    


# Execução Resultado:


# matt@matt-not:~/Workspace/rna-mlp$ python not-mlp-fakenews.py 

# xtr
# [[ 0.32047281  0.20490916  0.11518173 ... -0.31440875  0.05307668
#   -0.26900801]
#  [ 0.27560419  0.21750285 -0.0786874  ... -0.31240863 -0.13259882
#   -0.18418586]
#  [ 0.04351225  0.00626286  0.09158446 ... -0.16647205 -0.24911584
#    0.08957607]
#  ...
#  [-0.02578646 -0.08391939 -0.12484172 ...  0.17379841  0.25237963
#   -0.0236546 ]
#  [ 0.28096294  0.03672954 -0.08120567 ... -0.07374884  0.36524907
#   -0.25215837]
#  [ 0.29714856 -0.07268881 -0.65742952 ... -0.1127572   0.42861643
#   -0.20713833]]

# xte
# [[ 0.0400387   0.01445024 -0.009902   ...  0.00667283  0.0221667
#    0.00464787]
#  [ 0.25379452  0.64847648 -0.17045413 ... -0.27363336 -0.13846891
#    0.00517298]
#  [ 1.02346659  0.72193819 -0.65364176 ... -0.17876621  0.33173805
#   -0.38883775]
#  ...
#  [ 0.2522704   0.20023635 -0.25750679 ...  0.20530362  0.0487812
#   -0.27759078]
#  [-0.01036592  0.04693541 -0.02581567 ...  0.08582759  0.21091011
#   -0.10880172]
#  [ 0.36296281  0.01837418 -0.16082355 ...  0.23198058  0.38698336
#    0.5049004 ]]

# ytr
# [1. 0. 1. ... 0. 1. 0.]

# yte
# [1. 0. 0. ... 0. 1. 1.]

# Accuracy of Naive Bayes = 72.43%
# Accuracy of SVM = 88.27%
# Accuracy of Logistic Regression = 90.13%
