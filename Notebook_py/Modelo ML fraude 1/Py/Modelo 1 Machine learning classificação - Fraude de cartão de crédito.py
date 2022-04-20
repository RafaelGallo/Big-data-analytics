#!/usr/bin/env python
# coding: utf-8

# # Modelo 1 machine learning classificação - Fraude de cartão de crédito

# 
# # Definição do Problema de Negócio:  Fraude de cartão de crédito
# https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
# 
# # Definição do Problema
# 
# 
# É importante que as operadoras de cartão de crédito possamreconhecertransações fraudulentas no momento exato.
# Em que elas estiveremocorrendo, para que os clientes não sejam cobrados pelos itens que não compraram. 
# Nosso objetivo neste trabalho de análise é identificar umproblemacomumquando trabalhamos com dados que apresentam anomalias (fraudes, nessecaso). 
# Em cenários assim, temos uma situação comum que precisa ser tratada:
# Conforme você já sabe, usamos dados históricos para treinar modelosdeMachine Learning. 
# Esperamos que a operadora de cartão de crédito tenhamuitomais exemplos históricos de transações corretas do que transações fraudulentas.
# Se essa premissa não fosse verdadeira, a empresa já teria ido àfalência, concorda?. 
# Mas se entregarmos os dados dessa forma ao modelo de MachineLearning, ele vai aprender mais sobre uma categoria de transações o que outra. 
# Imaginepor exemplo que a empresa tenha essa massa de dados de umdia de transaçõesde cartão de crédito.
# 
# # Dados da empresa 
# 
# 25.000 exemplos de transações corretas (classe majoritária) 314 exemplos de transações fraudulentas (classe minoritária).
# Esse é tipicamente um problema de classificação, emque o modelo de Machine Learning deve analisar cada transação e classificar como fraudulenta ou não fraudulenta. 
# Cada modelo de Machine Learning procura pelo relacionamento matemático nos dados. 
# Mas nosso dataset está desbalanceado e, nessecaso, omodelo vai aprender muito mais sobre uma transação normal do que sobre uma transação fraudulenta. 
# Como resultado, o modelo pode classificar novastransações fraudulentas como se fossem transações normais, simplesmente porque aprendeu mais sobre uma classe do que sobre a outra. 
# Para minimizar esse problema, podemos aplicar uma de muitas técnicas de balanceamento de classes, criando dados sintéticos para aumentar transações fraudulentas (isso é chamado de oversampling) 
# Ou então podemos remover alguns registros da classe de transações normais (isso é chamadodeundersampling). 
# O undersampling é mais fácil, mas reduz o tamanho do dataset, oquenãoéo ideal. 
# O oversampling pode ser mais trabalhoso e mais complicado deexplicar, porém aumenta o tamanho do dataset criando dados sintéticos combaseemregras estatísticas e de forma aleatória, usando observações da classe minoritáriacomo ponto de partida. 
# Neste estudo dirigido usaremos uma técnica de balanceamento de classes chamada Randomly OverSampling Examples (ROSE) e comum pacote R perfeito para essa tarefa, chamado.....ROSE. 
# Dada a taxa de desequilíbrio de classe, o ideal é medir a precisãousandoamétrica Área Sob a curva Precision-Recall (AUPRC). 
# Usar apenas aacuráciadefinida pela matriz de confusão não é significativo para a classificaçãocomclasses desbalanceadas e também veremos isso.
# Contexto: Balanceamento de Classesem Dados de Fraudes Financeiras com ROSE(Random Over Sampling Examples)
# 

# # Base dados 
# 
# Usaremos o dataset público disponibilizado pelo Machine Learning Group. 
# O dataset deve ser baixado do link abaixo o dataset não será fornecidocomoscript, pois o arquivo é grande.
# 
# # Discrição do dataset
# O conjunto de dados contém transações realizadas comcartões decréditoem setembro de 2013 por portadores de cartões europeus. 
# Esse conjunto de dados apresenta transações que ocorreramemdoisdias, nas quais temos 492 fraudes em 284.807 transações. O conjunto dedadoséaltamente desequilibrado, a classe positiva (fraudes) representa 0,172 de todas transações. 
# Ele contém apenas variáveis de entrada numéricas que sãooresultadode uma transformação PCA. 
# Devido a problemas de confidencialidade, nãosepode fornecer os recursos originais e mais informações básicas sobreos dados. 
# Recursos V1, V2,… V28 são os principais componentes obtidos como PCA, osúnicos recursos que não foram transformados com o 
# 
# # PCA 
# **Tempo** 
# 
# **Valor** 
# 
# **Orecurso**
# 
# **Hora**
# 
# - Contém os segundos decorridos entre cada transação e aprimeiratransação no conjunto de dados. 
# - O recurso 'Valor' é o valor da transação. Orecurso 'Classe' é a variável de resposta e assume o valor 1 emcaso de fraudee0em caso contrário.
# 

# # Projeto
# 
# - Esse é um projeto prático da DSA na formação cientista de dados no primeiro módulo do curso Big Data Analytics com R e Microsoft Azure Machine Learning.
# 
# 
# **Objetivo: É encontrar possíveis cartões fraudes**
# 
# 
# #**Modelo ML**
# 
# - Decision tree
# - Naive bayes
# 
# # **Modelos hiperparâmetros**
# - Randomized SearchCV
# 
# - GridSearchCV	
# 

# # Importação das bibliotecas

# In[1]:


# Versão do python

from platform import python_version

print('Versão python neste Jupyter Notebook:', python_version())


# In[2]:


# Importação das bibliotecas 

import pandas as pd # Pandas carregamento csv
import numpy as np # Numpy para carregamento cálculos em arrays multidimensionais

# Visualização de dados
import seaborn as sns
import matplotlib as m
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly
import plotly.express as px

# Carregar as versões das bibliotecas
import watermark

# Warnings retirar alertas 
import warnings
warnings.filterwarnings("ignore")


# In[3]:


# Versões das bibliotecas

get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Versões das bibliotecas" --iversions')


# In[4]:


# Configuração para os gráficos largura e layout dos graficos

sns.set_palette("Accent")
sns.set(style="whitegrid", color_codes=True, font_scale=1.5)
color = sns.color_palette()


# # Base dados

# In[5]:


# Carregando a base de dados

df = pd.read_csv("creditcard.csv")
df


# In[6]:


# Exibido 5 primeiros dados
df.head()


# In[7]:


# Exibido 5 últimos dados 
df.tail()


# In[8]:


# Número de linhas e colunas
df.shape


# In[9]:


# Verificando informações das variaveis
df.info()


# In[10]:


# Exibido tipos de dados
df.dtypes


# In[11]:


# Total de colunas e linhas 

print("Números de linhas: {}" .format(df.shape[0]))
print("Números de colunas: {}" .format(df.shape[1]))


# In[12]:


# Exibindo valores ausentes e valores únicos

print("\nMissing values :  ", df.isnull().sum().values.sum())
print("\nUnique values :  \n",df.nunique())


# In[13]:


# Dados faltantes coluna óbitos

data = df[df["Class"].notnull()]
data.isna().sum()


# In[14]:


# Dados faltantes colunas internacoes

data = df[df["Class"].notnull()]
data.isna().sum()


# In[15]:


# Removendo dados ausentes do dataset 

df = df.dropna()
df.head()


# In[16]:


# Sum() Retorna a soma dos valores sobre o eixo solicitado
# Isna() Detecta valores ausentes

df.isna().sum()


# In[17]:


# Retorna a soma dos valores sobre o eixo solicitado
# Detecta valores não ausentes para um objeto semelhante a uma matriz.

df.notnull().sum()


# In[18]:


# Total de número duplicados

df.duplicated()


# In[19]:


# Renomeando estados por região 

df["Class"].unique()


# # Estatística descritiva

# In[20]:


# Exibindo estatísticas descritivas visualizar alguns detalhes estatísticos básicos como percentil, média, padrão, etc. 
# De um quadro de dados ou uma série de valores numéricos.

df.describe().T


# In[21]:


# Matriz correlação de pares de colunas, excluindo NA / valores nulos.

corr = df.corr()
corr


# In[22]:


# Gráfico da matrz de correlação 

plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), cmap="Blues")
plt.title("Correlations", size=16)
plt.show()


# In[23]:


# Gráfico da matriz de correlação

plt.figure(figsize=(50.5,30))
ax = sns.heatmap(corr, annot=True, cmap='YlGnBu');
plt.title("Matriz de correlação")


# # Análise de dados

# In[25]:


plt.figure(figsize=(20,10))
plt.title("Gráfico de barras")
sns.countplot(df["Class"])
plt.ylabel("Total")


# In[26]:


f, axes = plt.subplots(ncols=4, figsize=(24,4))

sns.boxplot(data=df, x="Class", y="V2", ax=axes[0])
axes[0].set_title("V2 vs Class Positive")

sns.boxplot(data=df, x="Class", y="V4", ax=axes[1])
axes[1].set_title("V4 vs Class Positive")

sns.boxplot(data=df, x="Class", y="V11", ax=axes[2])
axes[2].set_title("V11 vs Class Positive")

sns.boxplot(data=df, x="Class", y="V19", ax=axes[3])
axes[3].set_title("V19 vs Class Positive");

f, axes = plt.subplots(ncols=4, figsize=(24,4))

sns.boxplot(data=df, x="Class", y="V17", ax=axes[0])
axes[0].set_title("V17 vs Class Negetive")

sns.boxplot(data=df, x="Class", y="V14", ax=axes[1])
axes[1].set_title("V14 vs Class Negetive")

sns.boxplot(data=df, x="Class", y="V12", ax=axes[2])
axes[2].set_title("V12 vs Class Negetive")

sns.boxplot(data=df, x="Class", y="V10", ax=axes[3])
axes[3].set_title("V10 vs Class Negetive");


# In[27]:


plt.figure(figsize=(15, 8))

plt.pie(df.groupby('Class')['Class'].count(), labels=['Não Fraude', "Fraude"], autopct='%1.1f%%');
plt.title("Gráfico de pizza - Cartões fraudes")
plt.xlabel("Total")


# In[28]:


df.hist(bins = 40, figsize=(20.2, 20))
plt.show()


# # Pré - processamento de dados
# 
# - O processamento de dados começa com os dados em sua forma bruta e os converte em um formato mais legível (gráficos, documentos, etc.), dando-lhes a forma e o contexto necessários para serem interpretados por computadores e utilizados.
# 
# **Exemplo: Uma letra, um valor numérico. Quando os dados são vistos dentro de um contexto e transmite algum significado, tornam-se informações**

# In[29]:


# Defenindo base de treino e teste train e test

x = df.drop(["Class"], axis = 1)
y = df["Class"]


# In[30]:


# Visualizando linha e coluna da váriavel x
x.shape


# In[31]:


# Visualizando linha e coluna da váriavel y
y.shape


# # Escalonamento dados
# 
# Standard Scaler: padroniza um recurso subtraindo a média e escalando para a variância da unidade.
# A variância da unidade significa dividir todos os valores pelo desvio padrão. StandardScaler resulta em uma distribuição com um desvio padrão igual a 1 variância é igual a 1.
# 
# - Variância = desvio padrão ao quadrado.
# 
# - E 1 ao quadrado = 1.
# 
# - Standard Scaler torna a média da distribuição aproximadamente 0.

# In[32]:


from sklearn.preprocessing import StandardScaler

pre_scaler = StandardScaler()
x = pre_scaler.fit_transform(x)
x


# In[33]:


# Visualizando linhas e colunas do escalonamento x
x.shape


# In[34]:


# Visualizando linhas e colunas do escalonamento y
y.shape


# # Treino e teste
# 
# - Treino e teste do modelo machine learning 80 para dados de treino 20 para dados de teste
# 
# train_test_split: O train test split ele define o conjunto de dados de treinamento os dados em float deve estar entre 0.0 e 1 vai ser definirá o conjunto de dados teste.
# 
# - Test_size: E o tamanho do conjunto de teste para ser usando dados de teste 0.25 ou 25 por cento.
# 
# - Random_state: Devisão dos dados ele um objeto para controla a randomização durante a devisão dos dados

# In[36]:


# Treinando modelo machine learning e treino do modelo
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 123)


# In[37]:


# Total de linhas e colunas e linhas dos dados de treino x

x_train.shape


# In[38]:


# Total de linhas dos dados de treino y

x_train.shape


# In[39]:


# Total de linhas e colunas dos dados de treino x teste 

x_test.shape


# In[40]:


# Total de linhas e colunas dos dados de treino y teste 

y_test.shape


# # Modelo machine learning
# 
# **Modelo 01 - Decision Tree Classifier**

# In[41]:


# Modelo - Decision tree classifier

# Importação da biblioteca
from sklearn.tree import DecisionTreeClassifier

model_decision_tree = DecisionTreeClassifier(max_depth = 5) # Nome do algoritmo M.L
model_decision_tree_fit = model_decision_tree.fit(x_train, y_train) # Treinamento do modelo
model_decision_tree_scor = model_decision_tree.score(x_train, y_train) # Score do modelo dados treino x

print("Modelo - Decision Tree Classifier: %.2f" % (model_decision_tree_scor * 100)) # Score do modelo dados treino y


# In[42]:


# Previsão do modelo
model_decision_tree_pred = model_decision_tree.predict(x_test)
model_decision_tree_pred


# # Plot
# 
# - Plot da árvore

# In[43]:


# Gráfico da árvore
from sklearn import tree

fig, ax = plt.subplots(figsize=(65.5, 30), 
                       facecolor = "g")

tree.plot_tree(model_decision_tree, 
               ax = ax, 
               fontsize = 15, 
               rounded = True, 
               filled = True, 
               class_names = ["Não Fraude", "Fraude"])
plt.show()


# # Accuracy
# 
# - Ela indica performance geral do modelo dentros todos as classificações quantas modelo classificou corretamente.

# In[44]:


# Accuracy do modelo 
from sklearn.metrics import accuracy_score

accuracy_dt = accuracy_score(y_test, model_decision_tree_pred)
print("Acurácia - Decision Tree Classifier: %.2f" % (accuracy_dt * 100))


# # Matrix confusion ou Matriz de Confusão
# 
# A matriz de confusão uma tabela que indica erros e acertos do modelo comparando com um resultado.
# 
# - Verdadeiros Positivos: A classificação da classe positivo.
# - Falsos Negativos (Erro Tipo II): Erro em que o modelo previu a classe Negativo quando o valor real era classe Positivo;
# - Falsos Positivos (Erro Tipo I): Erro em que o modelo previu a classe Positivo quando o valor real era classe Negativo
# - Verdadeiros Negativos: Classificação correta da classe Negativo.

# In[45]:


# Matriz de confusão
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

matrix_confusion = confusion_matrix(y_test, model_decision_tree_pred)
matrix_confusion


# In[46]:


plt.figure(figsize=(10.5, 8))

ax = plt.subplot()
sns.heatmap(matrix_confusion, annot=True, ax = ax, fmt = ".1f", cmap="plasma"); 
ax.set_title('Confusion Matrix - Decision tree'); 
ax.xaxis.set_ticklabels(["Fraude", "Fraude Fraude"]); ax.yaxis.set_ticklabels(["Fraude", "Fraude Fraude"]);


# # Curva roc
# 
# - A curva roc ela exibir graficamente comparar a avaliar acurácia. As curvas roc integram três medidas precisão relacionadas a sensibilidade com os verdadeiro e positivo, especificidade com os verdadeiro negativo.

# In[47]:


# Cruva roc do modelo

from sklearn import metrics

roc_g = model_decision_tree.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  roc_g)
auc = metrics.roc_auc_score(y_test, roc_g)

plt.title("Curva roc - Decision Tree Classifier")
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# # Classification report
# 
# - O visualizador do relatório de classificação exibe as pontuações de precisão, recuperação, F1 e suporte para o modelo. Para facilitar a interpretação e a detecção de problemas, o relatório integra pontuações numéricas com um mapa de calor codificado por cores. Todos os mapas de calor estão na faixa para facilitar a comparação fácil de modelos de classificação em diferentes relatórios de classificação.

# In[48]:


# Classification report

from sklearn.metrics import classification_report

classification = classification_report(y_test, model_decision_tree_pred)
print("Modelo - Decision Tree Classifier")
print("\n")
print(classification)


# In[49]:


# Métricas do modelos

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

precision = precision_score(y_test, model_decision_tree_pred)
Recall = recall_score(y_test, model_decision_tree_pred)
Accuracy = accuracy_score(y_test, model_decision_tree_pred)
F1_Score = f1_score(y_test, model_decision_tree_pred)

precisao = pd.DataFrame({
    
    "Metricas" : ["precision",
                 "Recall", 
                  "Accuracy", 
                  "F1_Score"],
    
    "Resultado": [precision,
                Recall, 
                Accuracy, 
                F1_Score]})

precisao.sort_values(by = "Resultado", ascending = False)


# **Modelo 02 - Naive bayes**

# In[50]:


get_ipython().run_cell_magic('time', '', 'from sklearn.naive_bayes import GaussianNB\n\nmodel_naive_bayes = GaussianNB()\nmodel_naive_bayes_fit = model_naive_bayes.fit(x_train, y_train)\nmodel_naive_bayes_score = model_naive_bayes.score(x_train, y_train)\n\nprint("Modelo - Naive Bayes: %.2f" % (model_naive_bayes_score * 100))')


# In[51]:


# Previsão do modelo - Naive bayes

model_naive_bayes_pred_predict = model_naive_bayes.predict(x_test)
model_naive_bayes_pred_predict


# In[52]:


accuracy_nb = accuracy_score(y_test, model_naive_bayes_pred_predict)

print("Accuracy Naive bayes: %.2f" % (accuracy_nb * 100))


# In[53]:


matrix_confusion_3 = confusion_matrix(y_test, model_naive_bayes_pred_predict)
matrix_confusion_3


# In[54]:


plt.figure(figsize=(15, 8))

ax = plt.subplot()
sns.heatmap(matrix_confusion_3, annot=True, ax = ax, fmt = ".1f", cmap="plasma"); 
ax.set_title('Confusion Matrix - Naive bayes'); 
ax.xaxis.set_ticklabels(["Fraude", "Não Fraude"]); ax.yaxis.set_ticklabels(["Fraude", "Não Fraude"]);


# In[55]:


# Curva roc do modelo
from sklearn.metrics import roc_curve, roc_auc_score

roc = model_naive_bayes.predict_proba(x_test)[:,1]
tfp, tvp, limite = roc_curve(y_test, roc)
print('roc_auc', roc_auc_score(y_test, roc))

plt.subplots(1, figsize=(5,5))
plt.title('Curva ROC - Naive bayes')
plt.plot(tfp,tvp)
plt.xlabel('Especifidade')
plt.ylabel('Sensibilidade')
plt.plot([0, 1], ls="--", c = 'red')
plt.plot([0, 0], [1, 0], ls="--", c = 'green'), plt.plot([1, 1], ls="--", c = 'green')
plt.show()


# In[56]:


from sklearn.metrics import classification_report

classification = classification_report(y_test, model_naive_bayes_pred_predict)
print("Modelo")
print()
print(classification)


# In[57]:


precision = precision_score(y_test, model_naive_bayes_pred_predict)
Recall = recall_score(y_test, model_naive_bayes_pred_predict)
Accuracy = accuracy_score(y_test, model_naive_bayes_pred_predict)
F1_Score = f1_score(y_test, model_naive_bayes_pred_predict)

precisao = pd.DataFrame({
    
    "Metricas" : ["precision",
                 "Recall", 
                  "Accuracy", 
                  "F1_Score"],
    
    "Resultado": [precision,
                Recall, 
                Accuracy, 
                F1_Score]})

precisao.sort_values(by = "Resultado", ascending = False)


# In[ ]:





# In[58]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.linear_model import LogisticRegression\n\nmodel_regression_logistic = LogisticRegression()\nmodel_regression_logistic_fit = model_regression_logistic.fit(x_train, y_train)\nmodel_regression_logistic_score = model_regression_logistic.score(x_train, y_train)\n\nprint("Modelo - Regressão logistica: %.2f" % (model_regression_logistic_score * 100))')


# In[59]:


# Previsão do modelo
model_regression_logistic_pred = model_regression_logistic.predict(x_test)
model_regression_logistic_pred


# In[60]:


accuracy_regression_logistic = accuracy_score(y_test, model_regression_logistic_pred)

print("Accuracy -  Logistic regression: %.2f" % (accuracy_regression_logistic * 100))


# In[61]:


matrix_confusion_4 = confusion_matrix(y_test, model_regression_logistic_pred)
matrix_confusion_4


# In[62]:


plt.figure(figsize=(15, 8))

ax = plt.subplot()
sns.heatmap(matrix_confusion_4, annot=True, ax = ax, fmt = ".1f", cmap="plasma"); 
ax.set_title('Confusion Matrix - Regressão logistica'); 
ax.xaxis.set_ticklabels(["Fraude", "Não Fraude"]); ax.yaxis.set_ticklabels(["Fraude", "Não Fraude"]);


# In[63]:


roc = model_regression_logistic.predict_proba(x_test)[:,1]
tfp, tvp, limite = roc_curve(y_test, roc)
print('roc_auc', roc_auc_score(y_test, roc))

plt.subplots(1, figsize=(5,5))
plt.title('Curva ROC')
plt.plot(tfp,tvp)
plt.xlabel('Especifidade')
plt.ylabel('Sensibilidade')
plt.plot([0, 1], ls="--", c = 'red')
plt.plot([0, 0], [1, 0], ls="--", c = 'green'), plt.plot([1, 1], ls="--", c = 'green')
plt.show()


# In[64]:


classification = classification_report(y_test, model_regression_logistic_pred)
print("Modelo - Regressão logistica")
print()
print(classification)


# In[65]:


precision = precision_score(y_test, model_regression_logistic_pred)
Recall = recall_score(y_test, model_regression_logistic_pred)
Accuracy = accuracy_score(y_test, model_regression_logistic_pred)
F1_Score = f1_score(y_test, model_regression_logistic_pred)

precisao = pd.DataFrame({
    
    "Metricas" : ["precision",
                 "Recall", 
                  "Accuracy", 
                  "F1_Score"],
    
    "Resultado": [precision,
                Recall, 
                Accuracy, 
                F1_Score]})

precisao.sort_values(by = "Resultado", ascending = False)


# In[66]:


get_ipython().run_cell_magic('time', '', '\nfrom xgboost import XGBClassifier\n\nxgb = XGBClassifier()\nxgb_fit = xgb.fit(x_train, y_train)\nxgb_score = xgb.score(x_train, y_train)\nprint("Modelo - XGBoost: %.2f" % (xgb_score * 100))')


# In[67]:


# Previsão do modelo - XGBoost

xgb_pred = xgb.predict(x_test)
xgb_pred


# In[68]:


accuracy_XGBoost = accuracy_score(y_test, xgb_pred)
print("Accuracy - XGBoost: %.2f" % (accuracy_XGBoost * 100))


# In[69]:


matrix_confusion_5 = confusion_matrix(y_test, xgb_pred)
matrix_confusion_5


# In[70]:


plt.figure(figsize=(15, 8))

ax = plt.subplot()
sns.heatmap(matrix_confusion_5, annot=True, ax = ax, fmt = ".1f", cmap="plasma"); 
ax.set_title('Confusion Matrix - XGBOSST'); 
ax.xaxis.set_ticklabels(["Fraude", "Não Fraude"]); ax.yaxis.set_ticklabels(["Fraude", "Não Fraude"]);


# In[71]:


roc = xgb.predict_proba(x_test)[:,1]
tfp, tvp, limite = roc_curve(y_test, roc)
print('roc_auc', roc_auc_score(y_test, roc))

plt.subplots(1, figsize=(5,5))
plt.title('Curva ROC')
plt.plot(tfp,tvp)
plt.xlabel('Especifidade')
plt.ylabel('Sensibilidade')
plt.plot([0, 1], ls="--", c = 'red')
plt.plot([0, 0], [1, 0], ls="--", c = 'green'), plt.plot([1, 1], ls="--", c = 'green')
plt.show()


# In[72]:


classification = classification_report(y_test, xgb_pred)
print("Modelo 05 - XGBoost")
print()
print(classification)


# In[73]:


recision = precision_score(y_test, xgb_pred)
Recall = recall_score(y_test, xgb_pred)
Accuracy = accuracy_score(y_test, xgb_pred)
F1_Score = f1_score(y_test, xgb_pred)

precisao = pd.DataFrame({
    
    "Metricas" : ["precision",
                 "Recall", 
                  "Accuracy", 
                  "F1_Score"],
    
    "Resultado": [precision,
                Recall, 
                Accuracy, 
                F1_Score]})

precisao.sort_values(by = "Resultado", ascending = False)


# In[74]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.ensemble import GradientBoostingClassifier\n\nmodel_gradient_boosting = GradientBoostingClassifier()\nmodel_gradient_boosting_fit = model_gradient_boosting.fit(x_train, y_train)\nmodel_gradient_boosting_score = model_gradient_boosting.score(x_train, y_train)\n\nprint("Modelo - Gradient Boosting: %.2f" % (model_gradient_boosting_score * 100))')


# In[75]:


# Previsão do modelo - Gradient Boosting

model_gradient_boosting_pred = model_gradient_boosting.predict(x_test)
model_gradient_boosting_pred


# In[76]:


accuracy_model_gradient_boosting = accuracy_score(y_test, model_gradient_boosting_pred)

print("Acurácia - Gradient boosting: %.2f" % (accuracy_model_gradient_boosting * 100))


# In[77]:


matrix_confusion_6 = confusion_matrix(y_test, model_gradient_boosting_pred)
matrix_confusion_6


# In[78]:


plt.figure(figsize=(15, 8))

ax = plt.subplot()
sns.heatmap(matrix_confusion_6, annot=True, ax = ax, fmt = ".1f", cmap="plasma"); 
ax.set_title('Confusion Matrix - XGBOSST'); 
ax.xaxis.set_ticklabels(["Fraude", "Não Fraude"]); ax.yaxis.set_ticklabels(["Fraude", "Não Fraude"]);


# In[79]:


roc = model_gradient_boosting.predict_proba(x_test)[:,1]
tfp, tvp, limite = roc_curve(y_test, roc)
print('roc_auc', roc_auc_score(y_test, roc))

plt.subplots(1, figsize=(5,5))
plt.title('Curva ROC - Gradient boosting')
plt.plot(tfp,tvp)
plt.xlabel('Especifidade')
plt.ylabel('Sensibilidade')
plt.plot([0, 1], ls="--", c = 'red')
plt.plot([0, 0], [1, 0], ls="--", c = 'green'), plt.plot([1, 1], ls="--", c = 'green')
plt.show()


# In[80]:


classification = classification_report(y_test, model_gradient_boosting_pred)

print("Modelo - 06 - Gradient boosting")
print("\n")
print(classification)


# In[81]:


precision = precision_score(y_test, model_gradient_boosting_pred)
Recall = recall_score(y_test, model_gradient_boosting_pred)
Accuracy = accuracy_score(y_test, model_gradient_boosting_pred)
F1_Score = f1_score(y_test, model_gradient_boosting_pred)

precisao = pd.DataFrame({
    
    "Metricas" : ["precision",
                 "Recall", 
                  "Accuracy", 
                  "F1_Score"],
    
    "Resultado": [precision,
                Recall, 
                Accuracy, 
                F1_Score]})

precisao.sort_values(by = "Resultado", ascending = False)


# In[82]:


get_ipython().run_cell_magic('time', '', '\n# Importação da biblioteca \nfrom sklearn.ensemble import RandomForestClassifier\n\n# Modelo random forest classifier\nmodel_random_forest = RandomForestClassifier(max_depth = 2, random_state = 0)\n\n# Modelo treinamento \nmodel_random_forest_fit = model_random_forest.fit(x_train, y_train)\n\n# Modelo score do modelo\nmodel_random_forest_score = model_random_forest.score(x_train, y_train)\nprint("Score - Modelo random forest: %.2f" % (model_random_forest_score * 100))')


# In[83]:


# Previsão do modelo

model_random_forest_pred = model_random_forest.predict(x_test)
model_random_forest_pred


# In[84]:


# Accuracy do modelo
accuracy_random_forest = accuracy_score(y_test, model_random_forest_pred)

print("Acurácia - Random forest: %.2f" % (accuracy_random_forest * 100))


# In[85]:


# confusion Matrix

matrix_confusion_7 = confusion_matrix(y_test, model_random_forest_pred)
matrix_confusion_7


# In[86]:


# plot confusion Matrix

plt.figure(figsize=(15, 8))
ax = plt.subplot()

sns.heatmap(matrix_confusion_6, annot=True, ax = ax, fmt = ".1f", cmap="plasma"); 
ax.set_title('Confusion Matrix - Random forest'); 
ax.xaxis.set_ticklabels(["Fraude", "Não Fraude"]); ax.yaxis.set_ticklabels(["Fraude", "Não Fraude"]);


# In[87]:


# Curva roc do modelo
roc = model_random_forest.predict_proba(x_test)[:,1]
tfp, tvp, limite = roc_curve(y_test, roc)
print('roc_auc', roc_auc_score(y_test, roc))

plt.subplots(1, figsize=(5,5))
plt.title('Curva ROC - Random Forest')
plt.plot(tfp,tvp)
plt.xlabel('Especifidade')
plt.ylabel('Sensibilidade')
plt.plot([0, 1], ls="--", c = 'red')
plt.plot([0, 0], [1, 0], ls="--", c = 'green'), plt.plot([1, 1], ls="--", c = 'green')
plt.show()


# In[88]:


# Classificação report do modelo
classification = classification_report(y_test, model_random_forest_pred)

print("Modelo - 07 - Random Forest")
print("\n")
print(classification)


# In[89]:


# Métricas do modelo
precision = precision_score(y_test, model_random_forest_pred)
Recall = recall_score(y_test, model_random_forest_pred)
Accuracy = accuracy_score(y_test, model_random_forest_pred)
F1_Score = f1_score(y_test, model_random_forest_pred)

precisao = pd.DataFrame({
    
    "Metricas" : ["precision",
                 "Recall", 
                  "Accuracy", 
                  "F1_Score"],
    
    "Resultado": [precision,
                Recall, 
                Accuracy, 
                F1_Score]})

precisao.sort_values(by = "Resultado", ascending = False)


# # Resultados

# In[91]:


# Resultados 

modelos = pd.DataFrame({
    
    "Modelos" :["Regressão logistica", 
                "Random Forest",
                "XGBoost", "Gradient boosting",
                "Decision Tree",
                "Naive Bayes"],

    "Acurácia" :[accuracy_nb, 
                      accuracy_random_forest, 
                      accuracy_dt,
                      accuracy_XGBoost,
                      accuracy_model_gradient_boosting,
                      accuracy_regression_logistic]})

modelos.sort_values(by = "Acurácia", ascending = True)


# In[92]:


# Salvando modelo Machine learning

import pickle    
    
with open('model_decision_tree_pred.pkl', 'wb') as file:
    pickle.dump(model_decision_tree_pred, file)


# # Modelos de hiperparametros - RandomizedSearchCV

# In[ ]:


from sklearn.model_selection import cross_validate

results = cross_validate(model_decision_tree, 
                         x_train, 
                         y_train, 
                         cv=5,
                         scoring=('accuracy'),
                         return_train_score=True)

print(f"Mean train score {np.mean(results['train_score']):.2f}")
print(f"Mean test score {np.mean(results['test_score']):.2f}")


# In[ ]:


# Modelo Decision Tree Classifier

modelo_arvore_cla = DecisionTreeClassifier()
modelo_arvore_cla_fit = modelo_arvore_cla.fit(x_train, y_train)
modelo_arvore_cla


# In[ ]:


from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

SEED = 123456

# Parametros do modelo
parametros = {
    "max_depth" : randint(1, 10),
    "min_samples_split" : randint(32, 129),
    "min_samples_leaf" : randint(32, 129),
    "criterion" : ["gini", "entropy"]
}

#parametros = RandomizedSearchCV_fit.best_index_
parametros

# Modelo 01 - Decision Tree Classifier
modelo_arvore_cla = DecisionTreeClassifier()
DTC = RandomizedSearchCV(modelo_arvore_cla, 
                         parametros, 
                         random_state = SEED, 
                         cv = 5, 
                         return_train_score = True, 
                         n_iter = 10, 
                         scoring ='accuracy')

# Randomized SearchCV treinamento
RandomizedSearchCV_fit = DTC.fit(x_train, y_train)
pred_randomized_search_cv = DTC.predict(x_train)

# Resultados 
results_RandomizedSearchCV = RandomizedSearchCV_fit.cv_results_
print(results_RandomizedSearchCV)
print()
#resut = results_RandomizedSearchCV['params'][parametros]
#print(resut)

# Modelo Decision Tree Classifier
modelo_arvore_cla = DecisionTreeClassifier()
modelo_arvore_cla_fit = modelo_arvore_cla.fit(x_train, y_train)
modelo_arvore_cla_pred = modelo_arvore_cla.predict(x_test)

print("Parametros - RandomizedSearchCV:", parametros)


# In[ ]:


# Previsão modelo
modelo_arvore_cla_pred = modelo_arvore_cla.predict(x_test)
modelo_arvore_cla_pred


# In[ ]:


# Accuracy do modelo
acuracia_1 = metrics.accuracy_score(y_train, pred_randomized_search_cv)

print("Acuracia - Randomized search cv: %.2f" % (acuracia_1 * 100))


# In[ ]:


# Confusion matrix
matrix_confusion_1 = confusion_matrix(y_test, modelo_arvore_cla_pred)
matrix_confusion_1

#plot_confusion_matrix(matrix_confusion_1, show_normed=True, colorbar=False, class_names=['raude', 'Não fraude']) 


# In[ ]:


plt.figure(figsize=(18, 8))

ax = plt.subplot()
sns.heatmap(matrix_confusion_1, annot=True, ax = ax, fmt = ".1f", cmap="plasma"); 
ax.set_title('Confusion Matrix - Randomized search cv'); 
ax.xaxis.set_ticklabels(["Fraude", "Fraude Fraude"]); ax.yaxis.set_ticklabels(["Fraude", "Fraude Fraude"]);


# In[ ]:


classification = classification_report(y_test, modelo_arvore_cla_pred)

print("Modelo -  RandomizedSearchCV")
print()
print(classification)


# In[ ]:


roc_g = modelo_arvore_cla.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  roc_g)
auc = metrics.roc_auc_score(y_test, roc_g)

plt.title("Curva roc - Decision Tree Classifier")
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[ ]:


# Métricas do modelos

precision = precision_score(y_test, modelo_arvore_cla_pred)
Recall = recall_score(y_test, modelo_arvore_cla_pred)
Accuracy = accuracy_score(y_test, modelo_arvore_cla_pred)
F1_Score = f1_score(y_test, modelo_arvore_cla_pred)

precisao = pd.DataFrame({
    
    "Metricas" : ["precision",
                 "Recall", 
                  "Accuracy", 
                  "F1_Score"],
    
    "Resultado": [precision,
                Recall, 
                Accuracy, 
                F1_Score]})

precisao.sort_values(by = "Resultado", ascending = False)


# # Modelo de hiperparametros - GridSearchCV

# In[ ]:


from sklearn.model_selection import GridSearchCV

# Parametros modelo e cálculo do gini e entropy
parametros = {
  "max_depth" : [3, 5],
  "min_samples_split" : [32, 64, 128],
  "min_samples_leaf" : [32, 64, 128],
  "criterion" : ["gini", "entropy"]
}

# Modelo Decision Tree Classifier
modelo_arvore_cla = DecisionTreeClassifier()
DTCG = GridSearchCV(model_decision_tree, parametros, cv = 5, return_train_score = True, scoring = "accuracy")
grid_fit = DTCG.fit(x, y)

# Resultados do modelo
results_GridSearchCV = grid_fit.cv_results_
parametros = grid_fit.best_index_
results_GridSearchCV["params"][parametros]

print(results_GridSearchCV)
print()
print(f"Mean train score {results_GridSearchCV['mean_train_score'][parametros]:.2f}")
print()
print(f"mean test score {results_GridSearchCV['mean_test_score'][parametros]:.2f}")


# In[ ]:


# Previsão 

grid_pred = DTCG.predict(x_test)
grid_pred


# In[ ]:


# Classification report modelo

classification = classification_report(y_test, grid_pred)
print("Modelo -  GridSearchCV")
print()
print(classification)


# In[ ]:


# Confusion matrix

matrix_confusion_2 = confusion_matrix(y_test, grid_pred)
matrix_confusion_2


# In[ ]:


plt.figure(figsize=(18, 8))

ax = plt.subplot()
sns.heatmap(matrix_confusion_2, annot=True, ax = ax, fmt = ".1f", cmap="plasma"); 
ax.set_title('Confusion Matrix - GridSearchCV'); 
ax.xaxis.set_ticklabels(["Fraude", "Fraude Fraude"]); ax.yaxis.set_ticklabels(["Fraude", "Fraude Fraude"]);


# In[ ]:


# Accuracy score

acuracia_2 = accuracy_score(y_test, grid_pred)
print("Acuracia - GridSearchCV: %.2f" % (acuracia_2 * 100))


# In[ ]:


# Curva roc do modelo

roc_g = DTCG.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  roc_g)
auc = metrics.roc_auc_score(y_test, roc_g)

plt.title("Curva roc - Grid Search CV")
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[ ]:


# Métricas do modelo

precision = precision_score(y_test, grid_pred)
Recall = recall_score(y_test, grid_pred)
Accuracy = accuracy_score(y_test, grid_pred)
F1_Score = f1_score(y_test, grid_pred)

precisao = pd.DataFrame({
    
    "Metricas" : ["precision",
                 "Recall", 
                  "Accuracy", 
                  "F1_Score"],
    
    "Resultado": [precision,
                Recall, 
                Accuracy, 
                F1_Score]})

precisao.sort_values(by = "Resultado", ascending = False)


# In[ ]:


# Resultados - Modelos machine learning

modelos = pd.DataFrame({
    
    "Models" :["Modelo - GridSearchCV", 
               "Modelo - Randomized SearchCV"],

    "Acurácia" :[acuracia_1, 
                 acuracia_2]})

modelos_2 = modelos.sort_values(by = "Acurácia", ascending = False)
modelos_2.to_csv("modelos_3.csv")
modelos_2


# In[ ]:




