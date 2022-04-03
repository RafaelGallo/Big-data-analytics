# Modelo machine lerning classificao 

# Obs: Caso tenha problemas com a acentuação, consulte este link:
# https://support.rstudio.com/hc/en-us/articles/200532197-Character-Encoding

setwd() # Nome da raiz do projeto
getwd()

# Definição do Problema de Negócio: Previsão de Ocorrência de Câncer de Mama
# http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

############ Definição do Problema ############


# É importante que as operadoras de cartão de crédito possamreconhecertransações fraudulentas no momento exato.
# Em que elas estiveremocorrendo, para que os clientes não sejam cobrados pelos itens que não compraram. 
# Nosso objetivo neste trabalho de análise é identificar umproblemacomumquando trabalhamos com dados que apresentam anomalias (fraudes, nessecaso). 
# Em cenários assim, temos uma situação comum que precisa ser tratada:
# Conforme você já sabe, usamos dados históricos para treinar modelosdeMachine Learning. 
# Esperamos que a operadora de cartão de crédito tenhamuitomais exemplos históricos de transações corretas do que transações fraudulentas.
# Se essa premissa não fosse verdadeira, a empresa já teria ido àfalência, concorda?. 
# Mas se entregarmos os dados dessa forma ao modelo de MachineLearning, ele vai aprender mais sobre uma categoria de transações o que outra. 
# Imaginepor exemplo que a empresa tenha essa massa de dados de umdia de transaçõesde cartão de crédito.

############ Dados da empresa ############

# 25.000 exemplos de transações corretas (classe majoritária)
# 314 exemplos de transações fraudulentas (classe minoritária)
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

# Instalando bibliotecas
#install.packages("Rtsne")
#install.packages("DMwR")
#install.packages("ROSE")
#install.packages("Rborist")
#install.packages("xgboost")


############ Bibliotecas ############

library(ranger)
library(caret)
library(data.table)
library(ggplot2)
library(dplyr)
library(stringr)
library(caret) 
library(caTools)
library(ggplot2) 
library(corrplot) 
library(Rtsne) 
library(DMwR) 
library(ROSE)
library(rpart)
library(Rborist)
library(xgboost)

############ Base dados ############

# Usaremos o dataset público disponibilizado pelo Machine Learning Group. 
# O dataset deve ser baixado do link abaixo o dataset não será fornecidocomoscript, pois o arquivo é grande.

# Discrição do dataset
# O conjunto de dados contém transações realizadas comcartões decréditoem setembro de 2013 por portadores de cartões europeus. 
# Esse conjunto de dados apresenta transações que ocorreramemdoisdias, nas quais temos 492 fraudes em 284.807 transações. O conjunto dedadoséaltamente desequilibrado, a classe positiva (fraudes) representa 0,172 de todas transações. 
# Ele contém apenas variáveis de entrada numéricas que sãooresultadode uma transformação PCA. 
# Devido a problemas de confidencialidade, nãosepode fornecer os recursos originais e mais informações básicas sobreos dados. 
# Recursos V1, V2,… V28 são os principais componentes obtidos como PCA, osúnicos recursos que não foram transformados com o 

# PCA 
# 'Tempo' 
# 'Valor' 
# 'Orecurso'
# 'Hora' 

# Contém os segundos decorridos entre cada transação e aprimeiratransação no conjunto de dados. 
# O recurso 'Valor' é o valor da transação. Orecurso 'Classe' é a variável de resposta e assume o valor 1 emcaso de fraudee0em caso contrário.

###### 1.0 Base dados
data <- read.csv("creditcard.csv")
data

# Visualizando os 5 primieros dados
head(data)

# Visualizando os 5 últimos dados
tail(data)

# Visulizando colunas
str(data)

# linhas e colunas e linhas
dim(data)

# Resumo dados
str(data)

# Média de tendência central
summary(data$V4)
summary(data$V4)


############ Análise dados ############


###### 1.2 Análise exploratória de dados para variáveis numéricas

# Média
mean(data$V4)

# Médiana
median(data$V4)

# Quatil primeiro
quantile(data$V4)

# Segundo quatil
quantile(data$V4, probs = c(0.01, 0.99))

# Terceiro quatil
quantile(data$V4, seq(from = 0, to = 1, by = 0.20))

# Range
IQR(data$V4)
range(data$V4)
diff(range(data$V4))

# Correlação dos dados
corr_data <- cor(data)
corr_data

# Plot correplot
library(corrplot)
corrMat <- cor(corr_data[,-65])
corrplot(corrMat, method = "color", type = "lower", tl.cex = 0.8, tl.col = rgb(0,0,0))

# Gráfico de barras
common_theme <- theme(plot.title = element_text(hjust = 0.5, face = "bold"))
ggplot(data = data, aes(x = factor(Class), 
                      y = prop.table(stat(count)), fill = factor(Class),
                      label = scales::percent(prop.table(stat(count))))) +

geom_bar(position = "dodge") +
  geom_text(stat = 'count',
            position = position_dodge(.9),
            vjust = -0.5,
            size = 3) + 

scale_x_discrete(labels = c("Não fraude", "Fraude"))+
scale_y_continuous(labels = scales::percent)+
labs(x = 'Classe',
     y = 'Porcetagem') +
ggtitle("Distribuição de rótulos de classe") +
common_theme

# Gráfico boxplot
fig(14, 8)
ggplot(data, aes(x = factor(Class), y = Amount)) + 
  geom_boxplot() + 
  labs(x = 'Classe', 
       y = 'Montante') +
  ggtitle("Distribuição do valor da transação por classe") + 
  common_theme


# Gráfico de boxplot
boxplot(data$V4,
        main = "Valor do cartão",
        ylab = "Total",
        xlab = "Cartão V4")

# Gráfico histrograma
hist(data$V4,
     main = "Valor do cartão V4",
     ylab = "Total",
     xlab = "Cartão V4")

# Gráfico scatterplot dos cartãoes
plot(x = data$V4, y = data$V1,
     main = "Scatterplot - Cartão 1 e cartão 2",
     xlab = "Cartões 1 e 2",
     ylab = "Total")

# Medidas de Dispersão
# Ao interpretar a variância, números maiores indicam que 
# os dados estão espalhados mais amplamente em torno da 
# média. O desvio padrão indica, em média, a quantidade 
# de cada valor diferente da média.

var(data$V4)
sd(data$V4)
var(data$V4)
sd(data$V4)

###### 1.3 Análise Exploratória de Dados Para Variáveis Categóricas ###### 

# Criando tabelas de contingência - representam uma única variável categórica
# Lista as categorias das variáveis nominais

str(data)
table(data$Amount)
table(data$Amount)
str(data)

# Calculando a proporção de cada categoria

modelo_tabela <- table(data$Amount)
prop.table(modelo_tabela)

# Arrendondando os valores
modelo_tabela <- table(data$Amount)
modelo_tabela <- prop.table(modelo_tabela) * 100
round(modelo_tabela, digits = 1)

# Criando uma nova variável indicando "AMOUNT" conservadoras 
# (que as pessoas compram com mais frequência)
head(data)
data$AMOUNT_1 <- data$Amount %in% c("0", "1")
head(data)

# Verificando a nova variável
table(data$AMOUNT_1)

# Verificando o relacionamento entre 2 variáveis categóricas
# Criando uma crosstable 
# Tabelas de contingência fornecem uma maneira de exibir 
# as frequências e frequências relativas de observações 
# (lembra do capítulo de Estatística?), que são classificados 
# de acordo com duas variáveis categóricas. Os elementos de 
# uma categoria são exibidas através das colunas; 
# os elementos de outra categoria são exibidas sobre as linhas.

#install.packages("gmodels")
library(gmodels)

CrossTable(x = data$Amount,
           y = data$AMOUNT_1)

###### 1.4 Teste do Qui-quadrado ######


# Qui Quadrado, simbolizado por χ2 é um teste de hipóteses que se destina a encontrar um valor da dispersão 
# Para duas variáveis nominais, avaliando a associação existente entre variáveis qualitativas.
# É um teste não paramétrico, ou seja, não depende dos parâmetros populacionais, como média e variância.
# O princípio básico deste método é comparar proporções, isto é, as possíveis divergências 
# Entre as frequências observadas e esperadas para um certo evento.
# Evidentemente pode-se dizer que dois grupos se comportam de forma semelhante se as diferenças entre as frequências observadas 
# E as esperadas em cada categoria forem muito pequenas, próximas a zero.
# Ou seja, Se a probabilidade é muito baixa, ele fornece fortes evidências de que as duas variáveis estão associadas.

CrossTable(x = data$Amount, y = data$AMOUNT_1, chisq = TRUE)
chisq.test(x = data$Amount, y = data$AMOUNT_1)


###### 1.5 Medidas de Tendência Central ######

# Detectamos um problema de escala entre os dados, que então precisam ser normalizados
# O cálculo de distância feito pelo kNN é dependente das medidas de escala nos dados de entrada.
summary(data[c("Amount", "Class", "V10")])

###### 1.6 - Pré processamento ###### 

# Excluindo a coluna ID
# Independentemente do método de aprendizagem de máquina, deve sempre ser excluídas 
# variáveis de ID. Caso contrário, isso pode levar a resultados errados porque o ID 
# pode ser usado para unicamente "prever" cada exemplo. Por conseguinte, um modelo 
# que inclui um identificador pode sofrer de superajuste (overfitting), 
# e será muito difícil usá-lo para generalizar outros dados.

data$Time = NULL
data

###### 1.7 - Preparação de dados ###### 

# Removendo coluna "Time"
data <- data[,-1]

# Alterar a variável 'Class' para fator
data$Class <- as.factor(data$Class)
levels(data$Class) <- c("No_fraud", "Fraud")

# Verificando a coluna
table(data$Class)

###### 1.8 - Variáveis numéricas de escala ######
data[,-30] <- scale(data[,30])
head(data)

###### 1.9 - Criando uma função de normalização ###### 
normaliza <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Testando a função de normalizadora -> Os resultados deve ser indênticos
normaliza(c(1,2,3,4,5))
normaliza(c(0,10,20,30,40,50))

# Normalizando os dados do dataset
norm_data <- as.data.frame(lapply(data[1:31], normaliza))
head(data)

###### 2.0 - Criando o modelo machine learning

# Carregando biblioteca 
library(class)
library(caret)

# Criando dados de treino e teste
set.seed(1)
split <- sample.split(data$Class, SplitRatio = 0.7)
train <- subset(data, split == T)
teste <- subset(data, split == F)

# -> Escolhendo a técnica de amostragem
# relação de classe inicialmente
table(train$Class)
table(teste$Class)

# Downsampling
set.seed(9560)
treino_x <- downSample(x = treino[, -ncol(treino)],
                         y = treino$Class)
table(treino_x$Class)

# Upsampling
set.seed(9560)
upet_treino <- upSample(x = treino[, -ncol(treino)],
                     y = treino$Class)
table(upet_treino$Class)

# Rose
set.seed(9560)
traino_rose <- ROSE(Class ~ ., data  = treino)$data 
table(traino_rose$Class)

###### 2.1 - Modelo machine learning  ######
# Decision Tree Model
library(rpart)
library(rpart.plot)

# Nome do modelo
modelo_decision_tree <- rpart(Class ~ ., data = train, method = "class", minbucket = 20)
modelo_decision_tree

# Previsão do modelo
modelo_pred <- predict(modelo_decision_tree, data, type = "class")
modelo_pred

# Probabilidade
modelo_prob <- predict(modelo_decision_tree, data, type = "prob")
modelo_prob

# Gráfico da árvore
rpart.plot(modelo_decision_tree)

# Summario
summary(modelo_decision_tree)

# Confusion Matrix
table(modelo_pred, teste$Class)

# ROC
roc.curve(teste$Class, modelo_pred, plotit = TRUE)

# Modelo regressão logistica
# Regressão logistica
modelo_rl <- glm(Class ~ ., data = train, family = "binomial", control = list(maxit = 50))
modelo_rl

# Summario
summary(modelo_rl)

# Previsão do modelo
modelo_rl_pred <- predict(modelo_rl, newdata = teste, type = 'response')
modelo_rl_pred

# ROC
roc.curve(teste$Class, modelo_rl_pred, plotit = TRUE, main = "ROC - Regressão logistica")

# Confusion Matrix
table(modelo_rl_pred, test$Class)

######## Model 3 - Modelo ANN  Artificial Neural Network ########

library(neuralnet)

# Nome da rede neural
modelo_ANN = neuralnet(Class~., train, linear.output = FALSE)
modelo_ANN

# Gráfico ANN
plot(modelo_ANN)

# Previsão
predANN = compute(modelo_ANN,test)
predANN

# Súmario
summary(modelo_ANN)

# Resultado
resultANN = predANN$net.result
resultANN = ifelse(resultANN>0.5,1,0)
resultANN

# Matrix confusion
table(predANN, test$Class)

# Curva roc
roc = predict(modelo_ANN, newdata = test, n.trees = gbm.iter)
roc_auc = roc(test$Class, roc, plot = TRUE, col = "red")

######## Model 4 - Random Forest ########
model_random_forest = rpart(Class~., data = train, control = rpart.control(cp = .0005))
model_random_forest

# Predict - model
model_random_forest_pred <- predict(model_random_forest, test, rpart.type='class')
model_random_forest_pred

# Matrix confusion
table(model_random_forest_pred, test$Class)

# Percentual de previsões corretas com dataset de teste
mean(model_random_forest_pred==test$Class)

library(forecast)
accuracy(model_random_forest)

library(pROC)

# Curva roc
roc = predict(model_random_forest, newdata = test, n.trees = gbm.iter)
roc_auc = roc(test$Class, roc, plot = TRUE, col = "red", main =" Curva roc - Random Forest")

######## Model 6 - Naive bayes ########
library(e1071)
library(caTools)
library(caret)

# Modelo
model_naive_bayes <- naiveBayes(Class ~ ., data = train, usekernel = T)
model_naive_bayes

# Sumario dos dados
summary(model_naive_bayes)

# Pevisão modelo
model_naive_bayes_pred <- predict(model_naive_bayes, train, probability = TRUE)
model_naive_bayes_pred

# # Pevisão modelo segunda
model_nb_pred <- predict(model_naive_bayes, newdata = test)
model_nb_pred

# Matrix confusion
table(model_nb_pred, test$Class)

# Curva roc
roc = predict(model_naive_bayes, newdata = test, n.trees = gbm.iter)
roc_auc = roc(test$Class, roc, plot = TRUE, col = "red", main ="Roc Naive bayes") 

