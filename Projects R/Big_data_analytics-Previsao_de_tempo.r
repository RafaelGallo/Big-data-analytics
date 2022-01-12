# Big Data na Prática 1 - Analisando a Temperatura Média nas Cidades Brasileiras

# Configurando o diretório de trabalho
# Coloque entre aspas o diretório de trabalho que você está usando no seu computador
# Não use diretórios com espaço no nome
setwd()
getwd()

# Base dados nesse projeto
# Link base dados - http://berkeleyearth.org/data
# Link da base dados clima - https://drive.google.com/open?id=1nSwP3Y0V7gncbnG_DccNhrTRxmUNqMqa



# Carregando as bibliotecas
library(ggplot2)
library(scales)
library(dplyr)
library(readr)
library(data.table)


# Lendo o dataset 
# Leando arquivo read_csv2()
system.time(data_1 <- read.csv("TemperaturasGlobais.csv"))

# Leando arquivo em read.table()
system.time(data_2 <- read.table("TemperaturasGlobais.csv"))

# Leando arquivo em fread
system.time(base <- fread("TemperaturasGlobais.csv"))


# Fazendo base nova um subsets de dados carregados
df_city <- subset(data, Country == "Brazil")
df_city <- na.omit(df_city)

# Exibindo os 5 primeiros linhas
head(df_city)
View(df_city)

# Retornam o número de linhas ou colunas presentes
nrow(data)
nrow(df_city)

# Recuperar ou defina a dimensão de um objeto.
dim(df_city)


# Preparação dos dados e organização
# Covertendo datas do dataset
df_city$dt <- as.POSIXct(df_city$dt, format="%Y-%m-%d")
df_city$Month <- month(df_city$dt)
df_city$Year <- year(df_city$dt)


# Lendo base dados subsets 1 - Cidade Palmas
data_1 <- subset(df_city, City == "Palmas")
data_1 <- subset(data_1, Year %in% c(1796,1846,1896,1946,1996,2012))
data_1

# Lendo base dados subsets 2 - Cidade Curitiba
data_2 <- subset(df_city, City == "Curitiba")
data_2 <- subset(data_2, Year %in% c(1796,1846,1896,1946,1996,2012))
data_2

# Lendo base dadis subsets 3 - Cidade Recife
data_3 <- subset(df_city, City=='Recife')
data_3 <- subset(data_3,Year %in% c(1796,1846,1896,1946,1996,2012))               
data_3

# Construindo os Plots
plot_data_1 <- ggplot(data_1, aes(x = (Month), y = AverageTemperature, color = as.factor(Year))) +
  geom_smooth(se = FALSE, fill = NA, size = 2) +
  theme_light(base_size = 20) +
  xlab("")+
  ylab("") +
  scale_color_discrete("") +
  ggtitle("") +
  theme(plot.title = element_text(size = 18))

plot_data_2 <- ggplot(data_2, aes(x = (Month), y = AverageTemperature, color = as.factor(Year))) +
  geom_smooth(se = FALSE, fill = NA, size = 2) +
  theme_light(base_size = 20) +
  xlab("")+
  ylab("") +
  scale_color_discrete("") +
  ggtitle("") +
  theme(plot.title = element_text(size = 18))

plot_data_3 <- ggplot(data_3, aes(x = (Month), y = AverageTemperature, color = as.factor(Year))) +
  geom_smooth(se = FALSE, fill = NA, size = 2) +
  theme_light(base_size = 20) +
  xlab("")+
  ylab("") +
  scale_color_discrete("") +
  ggtitle("") +
  theme(plot.title = element_text(size = 18))

# Imprimindo na tela
plot_data_1
plot_data_2
plot_data_3

