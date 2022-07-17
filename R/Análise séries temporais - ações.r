# Big Data Analytics - Análise de Séries Temporais prevendo valor da ação VALE 

# Obs: Caso tenha problemas com a acentuação, consulte este link:
# https://support.rstudio.com/hc/en-us/articles/200532197-Character-Encoding
# Configurando o diretório de trabalho
# Coloque entre aspas o diretório de trabalho que você está usando no seu computador
# Não use diretórios com espaço no nome
setwd()
getwd()

# http://www.quantmod.com


# Instalando as bibliotecas
install.packages("quantmod")
install.packages("xts")
install.packages("readxl")
install.packages("dynlm")
install.packages("car")
install.packages("lmtest")
install.packages("fpp2")
install.packages("tseries")
install.packages("forecast")
install.packages("dygraphs")

# Carregando a bibliotecas
library(quantmod)
library(xts)
library(moments)
library(readxl) 
library(foreign)
library(dynlm) 
library(car) 
library(lmtest) 
library(sandwich)
library(fpp2) 
library(tseries) 
library(zoo)
library(forecast) 
library(ggplot2)

# Período da análise das ações
startDate = as.Date("2021-08-01")
endDate = as.Date("2022-01-12")

# Download base de dados do período
# Base de dados - Vale Mineradora
getSymbols("VALE", src="yahoo", from=startDate, to = endDate, auto.assign = T)

# Observando tipo dado retornor
class(VALE)
is.xts(VALE)

# Exibindo os dados das ações da Vale
head(VALE)
tail(VALE)


#### Análise dados ####

# Análisando dados - Fechado
VALE.Close <- VALE[, "VALE.Close"]
is.xts(VALE.Close)
head(Cl(VALE), 20)

# Analisando dados - Aberto
VALE.High <- VALE[, "VALE.High"]
is.xts(VALE.High)
head(Cl(VALE), 20)

# Gráfico - Candlestick dados total 
candleChart(VALE)

# Gráfico fechamento
plot(VALE.Close,
     main = "Ações da Vale - Fechada",
     col = "red",
     xlab = "Data",
     ylab = "Valor da ação",
     major.ticks = "months",
     minor.ticks = FALSE)

# Gráfico Open
plot(VALE.High,
     main = "Ações da Vale - Alta",
     col = "red",
     xlab = "Data",
     ylab = "Valor da ação",
     major.ticks = "months",
     minor.ticks = FALSE)

# Gráfico com a média 20 períodos, 2 desvios
# O desvio padrão com medida de volatidade
addBBands(n = 20, sd = 2)

# Indicador ADX com média 11 tipo exponencial
addADX(n = 11, maType = "EMA")

# Calculo dos logs diários
VALE.Ret <- diff(log(VALE.Close), lag = 1)

# Remove valores NA na prosição 1
VALE.Ret <- VALE.Ret[-1]

# Gráfico da taxa de retorno
plot(VALE.Ret,
     main = "Fechamento diário das ações da VALE",
     col = "red",
     xlab = "Data",
     ylab = "Retorno",
     major.ticks = "months",
     minor.ticks = FALSE)

# Calculando as medidas estatísticas
statNames <- c("Mean", 
               "Standard Deviation", 
               "Skewness", 
               "Kurtosis")
VALE.Stats <- c(mean(VALE.Ret),
                sd(VALE.Ret),
                skewness(VALE.Ret),
                kurtosis(VALE.Ret))
names(VALE.Stats)<-statNames
VALE.Stats

# Série em nível
VALE <- VALE
VALE.Close %>% ggtsdisplay(main = "Gráfico ACF, PACF")

VALE.Close %>% diff() %>% ggtsdisplay(main = "Série fechada em primeira diferença")

VALE.Close %>% diff(lag = 12) %>% ggtsdisplay(main = "Série fechada primeira diferença sazonal")

acf(VALE.Close)

pacf(VALE.Close)

# Ajuste sazonal
autoplot(VALE.Close)

VALE.Close %>% diff() %>% ggtsdisplay(main = "Série sazonal")

#### Modelo AR ####

# Modelo AR - Sazonalidade na ACF e indicacao de um ARIMA (4,1,1)(0,1,1)[12]
modelo_ar <- (fit <- Arima(VALE.Close, order = c(4, 1, 1), seasonal = c(0, 1, 1)))
modelo_ar

summary(modelo_ar)


autoplot(modelo_ar)

checkresiduals(modelo_ar)


library(dygraphs)
dygraph(VALE, main = " Valor das ações da VALE") %>% 
  dyAxis("x", drawGrid = TRUE) %>% dyEvent("2021-8-01", "2021", labelLoc = "bottom") %>% 
  dyEvent("2021-8-01", "2021", labelLoc = "bottom") %>% 
  dyEvent("2021-8-01", "2021", labelLoc = "bottom") %>% 
  dyEvent("2022-1-11", "2022", labelLoc = "bottom") %>%
  dyOptions(drawPoints = TRUE, pointSize = 2)

#### Modelo ARIMA ####
summary(VALE)

# Gráfico - Serie temporal
plot(VALE)


######### Modelo ARIMA 1 ######### 

model <- auto.arima(VALE.Close)
model

summary(model)

# Coeficientes p-values
p_values <- (1 - pnorm(abs(model$coef)/sqrt(diag(model$var.coef)))) * 2
p_values

#install.packages("FitAR")

library(FitAR)
LBQPlot(residuals(model), 36)

library(tseries)
jarque.bera.test(residuals(model))


######### Modelo ARIMA 2 - Automático ######### 

require(fpp2)
model_2 <- auto.arima(VALE.Close, stepwise = FALSE, approximation = FALSE)
model_2

summary(model_2)


# p_values coeficientes
pvalues <- (1 - pnorm(abs(model_2$coef)/sqrt(diag(model_2$var.coef)))) * 2
pvalues

# Checando resíduos
checkresiduals(model_2)

LBQPlot(residuals(model_2), 45)

jarque.bera.test(residuals(model_2))

autoplot(forecast(model_2, h = 25), title = "Previsão da ação fechada",
         xlab = "Total", ylab = "Valor")

# Previsão das ações 
model_3 <- predict(arima(VALE.Close, order = c(4,4,5)), n.ahead = 20)
pred <-model_3 $ pred
model_3 <- data.frame(pred)
model_3


###################### Modelo ARIMA 2 ######################

model_1<-arima(VALE.Close,order=c(1,1,0))
model_2<-arima(VALE.Close,order=c(1,1,1))
model_3<-arima(VALE.Close,order=c(1,1,2))
print(model_1);print(model_2);print(model_3)

plot(forecast(Arima(y = VALE.Close, order = c(1, 1, 2))))
plot(forecast(Arima(y = VALE.High, order = c(3, 3, 4))))

# Salvando os dados em um arquivo .rds
saveRDS(VALE, file = "VALE.rds")
df = readRDS("VALE.rds")
dir()
head(df)



