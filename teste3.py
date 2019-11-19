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


import time
execucaoInicio = time.time()


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
    data = data.iloc[0:50, [0, 1, 2, 3, 4]]

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

# print(xtr + xte)


X = np.concatenate((xtr, xte))
Y = np.concatenate((ytr, yte))



# split into 67% for train and 33% for test
X_train = xtr
X_test = xte
y_train = ytr
y_test = yte


# Cria o modelo
model = Sequential()
model.add(Dense(12, input_dim = 300, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

print(model.summary())

# Compilação do modelo
# Precisamos selecionar o otimizador que é o algoritmo específico usado para atualizar pesos enquanto 
# treinamos nosso modelo.
# Precisamos selecionar também a função objetivo que é usada pelo otimizador para navegar no espaço de pesos 
# (frequentemente, as funções objetivo são chamadas de função de perda (loss) e o processo de otimização é definido 
# como um processo de minimização de perdas).
# Outras funções aqui: https://keras.io/losses/
# A função objetivo "categorical_crossentropy" é a função objetivo adequada para predições de rótulos multiclass e 
# binary_crossentropy para classificação binária. 
# A métrica é usada para medir a performance do modelo. Outras métricas: https://keras.io/metrics/
# As métricas são semelhantes às funções objetivo, com a única diferença de que elas não são usadas para 
# treinar um modelo, mas apenas para avaliar um modelo. 
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Treinamento do modelo
# Epochs: Este é o número de vezes que o modelo é exposto ao conjunto de treinamento. Em cada iteração, 
# o otimizador tenta ajustar os pesos para que a função objetivo seja minimizada. 
# Batch_size: Esse é o número de instâncias de treinamento observadas antes que o otimizador execute uma 
# atualização de peso.
model.fit(X_train, y_train, epochs = 150, batch_size = 10)

# Avalia o modelo com os dados de teste
# Uma vez treinado o modelo, podemos avaliá-lo no conjunto de testes que contém novos exemplos não vistos. 
# Desta forma, podemos obter o valor mínimo alcançado pela função objetivo e o melhor valor alcançado pela métrica 
# de avaliação. Note-se que o conjunto de treinamento e o conjunto de teste são rigorosamente separados. 
# Não vale a pena avaliar um modelo em um exemplo que já foi usado para treinamento. 
# A aprendizagem é essencialmente um processo destinado a generalizar observações invisíveis e não a memorizar 
# o que já é conhecido.
loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Acurácia: %.2f%%" % (loss, accuracy*100))

# Gera as previsões
predictions = model.predict(X)

# Ajusta as previsões e imprime o resultado
rounded = [round(x[0]) for x in predictions]
print(rounded)
accuracy = np.mean(rounded == Y)
print("Acurácia das Previsões: %.2f%%" % (accuracy*100))



execucaoFim = time.time()
print(execucaoFim - execucaoInicio)
