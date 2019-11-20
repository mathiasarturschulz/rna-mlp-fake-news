import numpy as np
import re
import string
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import time


# Método responsável por realizar a limpeza do texto
def textClean(text):
    # Realiza a substituição de todos os caracteres diferentes do regex abaixo
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    # Coloca o texto em lowercase e realiza um slip no espaço
    text = text.lower().split()
    # Remove stopwords
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    # Atualiza a string sem os caracteres de pontuação
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


# Realiza a construção do array de sentenças que será utilizado no gensim doc2vec
def constructSentences(data):
    sentences = []
    for index, row in data.iteritems():
        # Converte para um formato legível para o computador
        sentences.append(TaggedDocument(utils.to_unicode(row).split(), [str(index)]))
    return sentences


# Método responsável por realizar o processamento dos textos
# Convertendo o texto para um formato numérico
def dataProcessing(data, vector_dimension=300):
    # Realiza a limpeza de cada registro
    for i in range(len(data)):
        data.loc[i, 'text'] = textClean(data.loc[i,'text'])

    # Realiza a construção das sentenças
    x = constructSentences(data['text'])
    y = data['label'].values

    # Modelo Doc2Vec
    model = Doc2Vec (
        min_count=1, 
        window=5, 
        vector_size=vector_dimension, 
        sample=1e-4, 
        negative=5, 
        workers=7, 
        epochs=10,
        seed=1
    )
    model.build_vocab(x)
    model.train(x, total_examples=model.corpus_count, epochs=model.iter)

    # Converte os dados numéricos para um array numpy
    x = np.zeros((len(model.docvecs), vector_dimension), dtype=float)
    for i in range(len(model.docvecs)):
        x[i] = model.docvecs[str(i)]

    return x, y


# Realiza a leitura do dataset e já realiza a remoção dos registros com campos vazios
def getDataset(path):
    data = pd.read_csv(path)
    # Pega as colunas 'text' e 'label'
    data = data.iloc[0:50, [3, 4]]
    # Dropa todas os registros que possuem campos vazios
    data.dropna(inplace = True)
    # Atualiza os index com as linhas removidas
    data = data.reset_index(drop = True)
    return data


# Criação do modelo MLP
def createModelMLP(vector_dimension = 300):
    # Criação do modelo
    model = Sequential()
    model.add(Dense(12, input_dim = vector_dimension, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(8, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    # Compilação do modelo
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model


# Método responsável por realizar o treinamento da MLP
def train(model, x_train, y_train, epochs = 150):
    # Treinamento do modelo
    model.fit(x_train, y_train, epochs = epochs, batch_size = 10)
    return model


# Método responsável por realizar a avaliação do modelo
def test(model, x_test, y_test, x, y):
    # Avalia o modelo com os dados de teste
    loss, accuracy = model.evaluate(x_test, y_test)
    print("\nLoss: %.2f, Acurácia: %.2f%%" % (loss, accuracy*100))

    # Gera as previsões
    predictions = model.predict(x)

    # Ajusta as previsões e imprime o resultado
    rounded = [round(x[0]) for x in predictions]
    print(rounded)
    accuracy = np.mean(rounded == y)
    print("Acurácia das Previsões: %.2f%%" % (accuracy*100))




PATH = './dataset/train.csv'
VECTOR_DIMENSION = 300
TEST_SIZE = 0.33
EPOCHS = 150

execucaoInicio = time.time()
print('### Fake News detection')

print('Lendo o dataset...')
data = getDataset(PATH)

print('Tratamento dos dados...')
x, y = dataProcessing(data, VECTOR_DIMENSION)

qtdRegistros = len(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33)

model = createModelMLP(VECTOR_DIMENSION)
model = train(model, x_train, y_train, EPOCHS)
model = test(model, x_test, y_test, x, y)

execucaoFim = time.time()
tempoExecucao = (execucaoFim - execucaoInicio) / 60


print('\n### Resultados: ')
print('Épocas: %i' % EPOCHS)
print('Dados de teste: %.2f%%' %(TEST_SIZE * 100))
print('Tempo de Execução: %.2f minutos' % tempoExecucao)
