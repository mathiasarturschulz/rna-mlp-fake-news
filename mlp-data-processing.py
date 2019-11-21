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
import matplotlib.pyplot as plt
from matplotlib import pyplot
from keras import backend
import codecs, json 
import os.path


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




PATH = './dataset/train.csv'
JSON_NAME = './dataset/processed-data-train.json'
VECTOR_DIMENSION = 300

execucaoInicio = time.time()
print('### Fake News detection - Tratamento dos dados')

print('Leitura do dataset...')
data = getDataset(PATH)

print('Tratamento dos dados...')
x, y = dataProcessing(data, VECTOR_DIMENSION)

print(x, y)



# Gravação dos dados tratados em um json
b = x.tolist() # nested lists with same data, indices
file_path = JSON_NAME ## your path variable
json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format

# # Gravação dos dados tratados em um json
# b = x.tolist() # nested lists with same data, indices
# file_path = JSON_NAME ## your path variable
# json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format


# Leitura
obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
b_new = json.loads(obj_text)
a_new = np.array(b_new)

print(a_new)



# # REALIZA A LEITURA DO JSON
# data = None
# with open('./data2.json') as json_file:
#     data = json.load(json_file)

# # GRAVA OS NOVOS VALORES
# newValue = {
#     'value': random.randint(1, 1000),
#     'date': datetime.today().strftime("%Y-%m-%d %H:%M:%S")
# }
# data['record'].append(newValue)
# # print(json.dumps(data, indent=4))
# print("Novos valores: ", newValue)

# # GRAVA O JSON COM OS NOVOS ARQUIVOS
# with open('./data2.json', 'w') as json_file:
#     json.dump(data, json_file)

# # Cálculo do tempo de execução
# execucaoFim = time.time()
# tempoExecucao = (execucaoFim - execucaoInicio) / 60


# print('\n### Resultados: ')
# print('QTD registros: %i ' % len(x))
# print('Épocas: %i' % EPOCHS)
# print('Dados de teste: %.2f%%' %(TEST_SIZE * 100))
# print('Tempo de Execução: %.2f minutos' % tempoExecucao)
