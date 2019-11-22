import re, string, time, codecs, json
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")


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


# Gravação dos dados tratados em um json
def writeJson(x, y, file_path):
    data = {
        'y': y.tolist(),
        'x': x.tolist()
    }
    json.dump(
        data, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4
    )




PATH = './dataset/train.csv'
JSON_NAME = './dataset/processed-data-train.json'
VECTOR_DIMENSION = 300

execucaoInicio = time.time()
print('### Fake News detection - Tratamento dos dados')

print('Leitura do dataset... ')
data = getDataset(PATH)

print('Tratamento dos dados... ')
x, y = dataProcessing(data, VECTOR_DIMENSION)

print('Gravando os dados tratatos no arquivo JSON... ')
writeJson(x, y, JSON_NAME)

# Cálculo do tempo de execução
execucaoFim = time.time()
tempoExecucao = (execucaoFim - execucaoInicio) / 60

print('Dados tratados e salvo no JSON com sucesso! ')
print('Tempo de execução do tratamento dos dados: %.2f minutos' % tempoExecucao)
