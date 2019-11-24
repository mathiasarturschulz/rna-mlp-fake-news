import codecs, json, os.path, sys, time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from matplotlib import pyplot
from keras import backend


# Realiza a leitura do arquivo json com o dados tratados
def readJson(file_path):
    if (not os.path.isfile(file_path)):
        return None, None
    
    json_string = codecs.open(file_path, 'r', encoding='utf-8').read()
    json_data = json.loads(json_string)
    
    # Transforma os dados json em um array numpy
    x = np.array(json_data['x'])
    y = np.array(json_data['y'])

    return x, y


# Método responsável por realizar o cálculo do RMSE
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis = -1))


# Criação do modelo MLP
def createModelMLP(vector_dimension = 300):
    # Variáveis auxiliares para serem apresentadas no final da execução
    # Quantidade de neurônio da camada de entrada
    input_layer_quantity_neuron = 12
    # Quantidade de neurônio das camadas intermediárias
    hidden_layer_quantity_neuron = 8
    # Quantidade de camadas intermediárias
    hidden_layer_quantity = 1

    # Criação do modelo

    # Camada de Entrada
    model = Sequential()
    model.add(Dense(
        input_layer_quantity_neuron, 
        input_dim = vector_dimension, 
        kernel_initializer = 'uniform', 
        activation = 'relu'
    ))

    # Camadas intermediárias
    model.add(Dense(
        hidden_layer_quantity_neuron, 
        kernel_initializer = 'uniform', 
        activation = 'relu'
    ))

    # Camada de saída
    model.add(Dense(
        1, 
        kernel_initializer = 'uniform', 
        activation = 'sigmoid'
    ))

    # Compilação do modelo
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', rmse, 'mape'])
    return model, input_layer_quantity_neuron, hidden_layer_quantity_neuron, hidden_layer_quantity


# Método responsável por realizar o treinamento da MLP
def train(model, x_train, y_train, epochs = 150):
    # Treinamento do modelo
    history = model.fit(x_train, y_train, epochs = epochs, batch_size = 10)
    return model, history


# Método responsável por realizar o teste do modelo
def test(model, x_test, y_test, x, y):
    # Avalia o modelo com os dados de teste
    loss, accuracy_model, rmse, mape = model.evaluate(x_test, y_test)
    
    # Gera as detecções se cada notícia é fake ou não
    detections = model.predict(x)

    # Ajusta as detecções
    rounded = [round(x[0]) for x in detections]
    accuracy_detection = np.mean(rounded == y)

    return loss, accuracy_model, rmse, mape, accuracy_detection




JSON_NAME = './dataset/processed-data-train.json'
VECTOR_DIMENSION = 300
TEST_SIZE = 0.33
EPOCHS = 150

execucaoInicio = time.time()
print('### Fake News detection - Treinamento e teste da MLP\n')

x, y = readJson(JSON_NAME)
if ((x is None) or (y is None)):
    print('Fim de Execução! ')
    print('Antes de treinar e testar a rede neural realize o tratamento dos dados! ')
    sys.exit()

# Divisão dos dados para treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33)

# Treinamento e teste do modelo
model, input_layer_neuron, hidden_layer_neuron, hidden_layer_quantity = createModelMLP(VECTOR_DIMENSION)
model, history = train(model, x_train, y_train, EPOCHS)
loss, accuracy_model, rmse, mape, accuracy_detection = test(model, x_test, y_test, x, y)

# Cálculo do tempo de execução
execucaoFim = time.time()
tempoExecucao = (execucaoFim - execucaoInicio) / 60


print("\n\n### Resultados: ")
print("Loss: %.2f" % loss)
print("R2: %.2f%%" % (accuracy_model * 100))
print("MAPE: %.2f" % mape)
print("RMSE: %.2f" % rmse)
print("R2 Detecções: %.2f%%" % (accuracy_detection * 100))
print("###")
print("QTD registros: %i " % len(x))
print("QTD registros treino: %i " % len(x_train))
print("QTD registros teste: %i " % len(x_test))
print("Dados de teste: %.2f%%" %(TEST_SIZE * 100))
print("QTD Épocas: %i" % EPOCHS)
print("QTD neurônios camada de entrada: %i" % input_layer_neuron)
print("QTD neurônios camadas intermediárias: %i" % hidden_layer_neuron)
print("QTD de camadas intermediárias: %i" % hidden_layer_quantity)
print("Tempo de Execução: %.2f minutos" % tempoExecucao)

# Apresentação dos gráficos
pyplot.plot(history.history['rmse'])
plt.title('Erro Médio Quadrático da Raiz - RMSE')
plt.xlabel('Épocas')
plt.ylabel('RMSE')
pyplot.show()

pyplot.plot(history.history['mape'])
plt.title('Erro Médio Percentual Absoluto - MAPE')
plt.xlabel('Épocas')
plt.ylabel('MAPE')
pyplot.show()

pyplot.plot(history.history['accuracy'])
plt.title('Coeficiente de Determinação - R2')
plt.xlabel('Épocas')
plt.ylabel('R2 Detecções')
pyplot.show()
