# Construindo um Modelo Preditivo para Análise de Risco

# Obs: Caso tenha problemas com a acentuação, consulte este link:
# https://support.rstudio.com/hc/en-us/articles/200532197-Character-Encoding

# Configurando o diretório de trabalho
# Coloque entre aspas o diretório de trabalho que você está usando no seu computador
setwd("F:/Meu Drive/Machine learning e deep learning (Nuvem)")
getwd()

# Importando as bibliotecas
library(ggplot2)
library(caret)
library(randomForest)
library(ROCR)

# Biblioteca de utilitários para construção de gráficos
source("plot_utils.R")

# 1.0 - Carregando dataset em dataframe
credit.df <- read.csv("credit_dataset.csv", header = TRUE, sep = ",")
head(credit.df)

# Visualizando os 5 primeiros dados
head(credit.df )

# Visualizando os 5 últimos dados
tail(credit.df )

# Visualizando linhas colunas
dim(credit.df )

# 2.0 #### Pré-processamento
# 1.0 convertendo variáveis para tipo fator - Categórica
to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}

# 1.2 Normalização dados
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center=T, scale=T)
  }
  return(df)
}

# 1.3 Normalização das variáveis
numeric.vars <- c("credit.duration.months", 
                  "age", 
                  "credit.amount")
credit.df <- scale.features(credit.df, numeric.vars)

# 1.4 Variáveis do tipo fator
categorical.vars <- c('credit.rating', 
                      'account.balance', 
                      'previous.credit.payment.status',
                      'credit.purpose', 
                      'savings', 
                      'employment.duration', 
                      'installment.rate',
                      'marital.status', 
                      'guarantor', 
                      'residence.duration', 
                      'current.assets',
                      'other.credits', 
                      'apartment.type', 
                      'bank.credits', 
                      'occupation', 
                      'dependents', 
                      'telephone', 
                      'foreign.worker')
credit.df <- to.factors(df = credit.df, variables = categorical.vars)
head(credit.df)

# 3.0 ########### Dividindo os dados em treino e teste - 60:40 ratio ###########
indexes <- sample(1:nrow(credit.df), size = 0.6 * nrow(credit.df))
train.data <- credit.df[indexes,]
test.data <- credit.df[-indexes,]

# 3.1 Feature Selection
# Função para seleção de variáveis
run.feature.selection <- function(num.iters=20, feature.vars, class.var){
  set.seed(10)
  variable.sizes <- 1:10
  control <- rfeControl(functions = rfFuncs, method = "cv", 
                        verbose = FALSE, returnResamp = "all", 
                        number = num.iters)
  results.rfe <- rfe(x = feature.vars, y = class.var, 
                     sizes = variable.sizes, 
                     rfeControl = control)
  return(results.rfe)
}

# 3.2 Executando a função
rfe.results <- run.feature.selection(feature.vars = train_data[,-1], class.var = train_data[,1])
rfe.results

# 3.3 Visualizando os resultados
rfe.results
varImp((rfe.results))

# 3.4 Criando e Avaliando o Modelo
## separate feature and class variables
test.feature.vars <- test.data[,-1]
test.class.var <- test.data[,1]

# 4.0 ########### Modelo machine learning ###########
# Modelo - Regressão logística

# 4.1 Formulas
formula.init <- "credit.rating ~ ."
formula.init <- as.formula(formula.init)

# 4.2 Modelo RL
lr.model <- glm(formula = formula.init, data = train.data, family = "binomial")
lr.model

# 4.3 Visualizando modelo 
summary(lr.model)

# 4.4 Testando o modelo nos dados de teste
lr.predictions <- predict(lr.model, test.data, type="response")
lr.predictions <- round(lr.predictions)
lr.predictions

# 4.5 Avaliando o modelo
confusionMatrix(table(data = lr.predictions, reference = test.class.var), positive = '1')

# 5.0 ########### Feature selection ###########
formula <- "credit.rating ~ ."
formula <- as.formula(formula)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
model <- train(formula, data = train.data, method = "glm", trControl = control)
importance <- varImp(model, scale = FALSE)
plot(importance)

# 5.1 Construindo o modelo com as variáveis selecionadas
formula.new <- "credit.rating ~ account.balance + 
                credit.purpose + 
                previous.credit.payment.status + 
                savings + 
                credit.duration.months"
formula.new <- as.formula(formula.new)
lr.model.new <- glm(formula = formula.new, data = train.data, family = "binomial")

# 5.2 Visualizando o modelo
summary(lr.model.new)

# 5.3 Testando o modelo nos dados de teste
lr.predictions.new <- predict(lr.model.new, test.data, type = "response") 
lr.predictions.new <- round(lr.predictions.new)

# 5.4 Avaliando o modelo
confusionMatrix(table(data = lr.predictions.new, reference = test.class.var), positive = '1')

# 6.0 ########### Avaliando a performance do modelo ###########

# 6.1 Criando curvas ROC
lr.model.best <- lr.model
lr.prediction.values <- predict(lr.model.best, test.feature.vars, type = "response")
predictions <- prediction(lr.prediction.values, test.class.var)
par(mfrow = c(1,2))
plot.roc.curve(predictions, title.text = "Curva ROC")
plot.pr.curve(predictions, title.text = "Curva Precision/Recall")
