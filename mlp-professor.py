
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy

#https://www.kaggle.com/uciml/pima-indians-diabetes-database

# Carregando o dataset
dataset = numpy.loadtxt("./Exemplo MLP/data.csv", delimiter=",")

# Imprime o dataset
print(dataset)

# Split em variáveis de input (X) e output (Y) 
X = dataset[:,0:8]
Y = dataset[:,8]

print(X)
print(Y)

# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

# Cria o modelo
model = Sequential()
model.add(Dense(12, input_dim = 8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.summary()

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
accuracy = numpy.mean(rounded == Y)
print("Acurácia das Previsões: %.2f%%" % (accuracy*100))
