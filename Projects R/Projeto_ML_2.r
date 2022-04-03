# Lista de Exercícios Parte 2 - Capítulo 11

# Obs: Caso tenha problemas com a acentuação, consulte este link:
# https://support.rstudio.com/hc/en-us/articles/200532197-Character-Encoding

# Configurando o diretório de trabalho
# Coloque entre aspas o diretório de trabalho que você está usando no seu computador
# Não use diretórios com espaço no nome
setwd("C:/FCD/BigDataRAzure/Cap12")
getwd()


# Regressão Linear
# Definição do Problema: Prever as notas dos alunos com base em diversas métricas
# https://archive.ics.uci.edu/ml/datasets/Student+Performance
# Dataset com dados de estudantes
# Vamos prever a nota final (grade) dos alunos

# Carregando o dataset
df <- read.csv2('estudantes.csv')
df

# Explorando os dados
head(df)
summary(df)
str(df)
any(is.na(df))

# install.packages("ggplot2")
# install.packages("ggthemes")
# install.packages("dplyr")
library(ggplot2)
library(ggthemes)
library(dplyr)

# Colunas númericas
col_numerica <- sapply(df, is.numeric)
col_numerica

# Fltro das colunas numéricas para correlação
corr_data <- cor(df[, col_numerica])
corr_data

# Gráfico da matriz de correlação
library(corrplot)
library(corrgram)

# Criando um corrplot
corrplot(corr_data, method = 'color')

# Criando um corrgram
corrgram(df)
corrgram(df, order = TRUE, lower.panel = panel.shade,
         upper.panel = panel.pie, text.panel = panel.txt)

# Gráfico de histograma
ggplot(df, aes(x = G3)) + 
  geom_histogram(bins = 20, 
                 alpha = 0.5, fill = 'blue') + 
  theme_minimal()

# Treinando e Interpretando o Modelo
# Import Library
library(caTools)

# Criando as amostras de forma randômica
set.seed(101)
amostra_p <- sample.split(df$age, SplitRatio = 0.70)
amostra_p

# ***** Treinamos nosso modelo nos dados de treino *****
# *****   Fazemos as predições nos dados de teste

# Criando dados de treino - 70% dos dados
train <- subset(df, amostra_p == TRUE)
train

# Criando dados de teste - 30% dos dados
teste <- subset(df, amostra_p == FALSE)
teste

# Modelo regressão linear
model_1 <- lm(G3 ~ ., train)
model_2 <- lm(G3 ~ G3 + G1, train)
model_3 <- lm(G3 ~ absences, train)
model_4 <- lm(G3 ~ Medu, train)

# Sumário dos modelos
summary(model_1) # 0.86
summary(model_2) # 0.82
summary(model_3) # 0.0002675
summary(model_4) # 0.06442

# Visualizando modelo fazendo as previsões
# Obtendo os resíduos
mod_res_1 <- residuals(model_1)
mod_res_1

mod_res_2 <- residuals(model_2)
mod_res_2

mod_res_3 <- residuals(model_3)
mod_res_3

mod_res_4 <- residuals(model_4)
mod_res_4

# Dados em dataframe
mod_res_1 <- as.data.frame(mod_res_1)
mod_res_1

mod_res_2 <- as.data.frame(mod_res_2)
mod_res_2

mod_res_3 <- as.data.frame(mod_res_3)
mod_res_3

mod_res_4 <- as.data.frame(mod_res_4)
mod_res_4


# Gráfico de histograma dos resíduos
ggplot(mod_res_1, aes(mod_res_1)) +  
  geom_histogram(fill = 'blue', 
                 alpha = 0.5, 
                 binwidth = 1,
                 main = "Gráfico dos resíduos",
                 xlab = "Resíduos")

# Previsão do modelo
model_1 <- lm(G3 ~., train)
model_pred <- predict(model_1, teste)
model_pred

# Gráfico do modelo regressão linear
plot(df$G3, pch = 20, cex = 1.5,
     col = "blue",
     main = "Gráfico regressão linear",
     ylab = "Total",
     xlab = "Idade")
abline(lm(df$G3 ~ teste))

# Visualizando os valores previstos e observados
resultados <- cbind(model_pred, teste$G3) 
colnames(resultados) <- c('Previsto','Real')
resultados <- as.data.frame(resultados)
resultados
min(resultados)

# Tratando os valores negativos
zero_t <- function(x){
  if (x < 0){
    return(0)
  }else{
    return(x)
  }
}

# Aplicando função para tratar valores negativos na previsão
resultados$Previsto <- sapply(resultados$Previsto, zero_t)
resultados$Previsto

# Calculando erro médio
# Quão distantes valores previstos estão dos valores observados

# MSE
MSE <- mean((resultados$Real - resultados$Previsto) ^ 2)
MSE

# RMSE
RMSE <- MSE^0.5
RMSE

# R Squared
R_SSE = sum((resultados$Previsto - resultados$Real))
R_SST = sum((mean(df$G3) - resultados$Real) ^ 2)
R_SSE
R_SST

# R-Squared
# Ajuda a avaliar o nível de precisão do nosso modelo. Quanto maior, melhor, sendo 1 o valor ideal.
R2 = 1 - (R_SSE/R_SST)
R2
