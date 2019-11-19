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

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

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




data = pd.read_csv('./fake-news-dataset/train.csv')

# data = data.iloc[1:10, [0,1,2,3,4]].values
# Pega os 100 primeiros registros
data = data.iloc[0:100, [0, 1, 2, 3, 4]]
# print(data[1:10, [1])
print(data)

missing_rows = []
for i in range(len(data)):
    if data.loc[i, 'text'] != data.loc[i, 'text']:
        missing_rows.append(i)
data = data.drop(missing_rows).reset_index().drop(['index','id'],axis=1)

for i in range(len(data)):
    data.loc[i, 'text'] = cleanup(data.loc[i,'text'])

x = constructLabeledSentences(data['text'])
y = data['label'].values
print('x')
print(x)
print('y')
print(y)
# print(x.values)


# text_model = Doc2Vec(min_count=1, window=5, vector_size=300, sample=1e-4, negative=5, workers=7, epochs=10,
#                         seed=1)

# print('text_model')
# print(text_model)

# text_model.build_vocab(x)
# text_model.train(x, total_examples=text_model.corpus_count, epochs=text_model.iter)

# train_size = int(0.8 * len(x))
# test_size = len(x) - train_size

# text_train_arrays = np.zeros((train_size, vector_dimension))
# text_test_arrays = np.zeros((test_size, vector_dimension))
# train_labels = np.zeros(train_size)
# test_labels = np.zeros(test_size)

# for i in range(train_size):
#     text_train_arrays[i] = text_model.docvecs['Text_' + str(i)]
#     train_labels[i] = y[i]

# j = 0
# for i in range(train_size, train_size + test_size):
#     text_test_arrays[j] = text_model.docvecs['Text_' + str(i)]
#     test_labels[j] = y[i]
#     j = j + 1






# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

# Cria o modelo
model = Sequential()
model.add(Dense(12, input_dim = 8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.summary()

# Compilação do modelo
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Treinamento do modelo
model.fit(X_train, y_train, epochs = 150, batch_size = 10)

# Avalia o modelo com os dados de teste
loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Acurácia: %.2f%%" % (loss, accuracy*100))

# Gera as previsões
predictions = model.predict(X)

# Ajusta as previsões e imprime o resultado
rounded = [round(x[0]) for x in predictions]
print(rounded)
accuracy = numpy.mean(rounded == Y)
print("Acurácia das Previsões: %.2f%%" % (accuracy*100))
