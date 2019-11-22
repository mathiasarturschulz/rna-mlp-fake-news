# rna-mlp-fake-news

### Dataset Fake News

https://www.kaggle.com/mohit28rawat/fake-news

### Executando a detecção se uma notícia é uma fake news ou não

##### 1º - Tratamento dos dados

Os textos devem ser convertidos em um formato numérico.

Execute o comando abaixo para realizar o tratamento do dataset.

```
python mlp-data-processing.py
```

##### 2º - Treinamento e avaliação da rede MLP

Execute o comando abaixo para treinar a rede com dados de treino, avaliar a rede com dados de teste e visualizar os resultados e gráficos gerados.

```
python mlp-fake-news.py
```

##### 1º - Exemplo execução - Tratamento dos dados

```
matt@matt-not:~/Workspace/rna-mlp$ python mlp-data-processing.py 
### Fake News detection - Tratamento dos dados
Leitura do dataset... 
Tratamento dos dados... 
Gravando os dados tratatos no arquivo JSON... 
Dados tratados e salvo no JSON com sucesso! 
Tempo de execução do tratamento dos dados: 4.23 minutos
```

##### 2º - Exemplo execução - Treinamento e avaliação da rede MLP

```
matt@matt-not:~/Workspace/rna-mlp$ python mlp-fake-news.py 
Using TensorFlow backend.
### Fake News detection - Treinamento e teste da MLP

-- apresenta a execução das épocas
-- apresenta os gráficos

### Resultados: 
Loss: 0.61
Acurácia: 90.56%
MAPE: 51179628.00
RMSE: 0.10
Acurácia Detecções: 95.92%
###
QTD registros: 20761 
QTD registros treino: 13909 
QTD registros teste: 6852 
Dados de teste: 33.00%
Épocas: 150
QTD neurônios camada de entrada: 12
QTD neurônios camadas intermediárias: 8
QTD de camadas intermediárias: 1
Tempo de Execução: 3.29 minutos
```
