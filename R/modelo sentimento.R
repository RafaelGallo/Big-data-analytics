# Mini-Projeto 1 - Análise de Sentimentos em Redes Sociais

# Este projeto é parte integrante do curso Big Data Analytics com R e Microsoft Azure da Formação Cientista de Dados. 
# O objetivo é captutar dados da rede social Twitter e realizar análise de sentimentos com os dados capturados. 
# Para que este projeto possa ser executado, diversos pacotes devem ser instalados e carregados.
# Todo o projeto será descrito de acordo com suas etapas. 
# Primeiro usaremos o cálculo de score de sentimento e em seguida usaremos um classificador com o algoritmo Naive Bayes

install.packages("twitteR")
install.packages("httr")
install.packages("knitr")
install.packages("rmarkdown")
install.packages("tm") # Instalando o pacote para Text Mining.
install.packages("SnowballC")
install.packages("RColorBrewer") # Instalando o pacote wordcloud
install.packages("wordcloud")

library(SnowballC)
library(tm)
library(twitteR)
library(httr)
library(knitr)
library(rmarkdown)
library(RColorBrewer)
library(wordcloud)

# Carregando a biblioteca com funções de limpeza
source('utils.R')
options(warn=-1)

###### Etapa 1 - Autenticação ###### 
# Abaixo você encontra o processo de autenticação. Lembre-se que você precisa ter uma conta criada no Twitter
# e criar uma aplicação. Os passos para criação da aplicação estão detalhados na especificação do projeto.


# Criando autenticação no Twitter
key <- "9GiNf3GH6LJJstwTZ35l8FvT5"
secret <- "d2DOy4mozbBX9v95KPAq9hMovxrCqzO7rzl0pweaLW3klbYLqV"
token <- "703383646602981377-M3SYWCKiY0ZdqnBlfFbLlvZPFCANCp5"
tokensecret <- "FY0bmCWM2d6KzJ7YUSUMrqgdlrarThr9DUQnsrKmN6Nw8"

# Autenticação. Responda 1 (Yes) quando perguntado sobre utilizar direct connection.
setup_twitter_oauth(key, secret, token, tokensecret)


###### Etapa 2 - Conexão ######
# Aqui vamos testar a conexão e capturar os tweets. Quanto maior sua amostra, mais precisa sua análise. 
# Mas a coleta de dados pode levar tempo, dependendo da sua conexão com a internet. Comece com 100 tweets, 
# pois à medida que você aumenta a quantidade, vai exigir mais recursos do seu computador. Buscaremos tweets com referência a hashtag #BigData.

# Verificando a timeline do usuário
userTimeline("dsacademybr")

###### Carregando dataset ######

library(readr)
data <- read_csv("G:/Meu Drive/Machine learning e deep learning/Cursos - Google Drive/DSA/Formação cientista de dados 2.0/Big Data Analytics com R e Microsoft Azure Machine Learning/ML Big data analytics - Projetos/Análise de Redes Sociais/Project r/Tweets.csv")
head(data)

# Visualizando os cinco primeiros dados
head(data)

# Visualizando os cinco últimos dados
tail(data)

# Linhas e colunas
dim(data)

###### Etapa 3 - Tratamento dos dados coletados através de text mining ######

# Aqui vamos instalar o pacote tm, para text mining. Vamos converter os tweets coletados em um objeto do
# tipo Corpus, que armazena dados e metadados e na sequência faremos alguns processo de limpeza, como
# remover pontuação, converter os dados para letras minúsculas e remover as stopwords (palavras comuns do idioma inglês, neste caso).


# Tratamento (limpeza, organização e transformação) dos dados coletados
tweetlist <- sapply(text, function(x) x$getText())
tweetlist <- iconv(tweetlist, to = "utf-8", sub="")
tweetlist <- limpaTweets(tweetlist)
tweetcorpus <- Corpus(VectorSource(tweetlist))
tweetcorpus <- tm_map(tweetcorpus, removePunctuation)
tweetcorpus <- tm_map(tweetcorpus, content_transformer(tolower))
tweetcorpus <- tm_map(tweetcorpus, function(x)removeWords(x, stopwords()))


###### Etapa 4 - Wordcloud, associação entre as palavras e dendograma ######

# Vamos criar uma nuvem de palavras (wordcloud) para verificar a relação entre as palavras que ocorrem com
# mais frequência. Criamos uma tabela com a frequência das palavras e então geramos um dendograma, que
# mostra como as palavras se relaiconam e se associam ao tema principal (em nosso caso, o termo BigData).


# Gerando uma nuvem palavras
pal2 <- brewer.pal(8,"Dark2")
wordcloud(data,
          min.freq = 2,
          scale = c(5,1),
          random.color = F,
          max.word = 60,
          random.order = F,
          colors = pal2)


#-----------------------------------------------------------------------
# Preprocessamento do texto para análise de sentimentos.

# Faz o preproceamento padrão do texto.
data$text <- data$text %>%
  str_replace("Opinião Geral:", "") %>%   # Remove começo.
  str_to_lower() %>%                      # Caixa baixa.
  str_replace_all(" *-+ *", "") %>%       # Remove hífen.
  str_replace_all("[[:punct:]]", " ") %>% # Pontuação por espaço.
  removeNumbers() %>%                     # Remove números.
  trimws()                                # Sem espaços nas bordas.

# Stop words padrão do idioma português.
stopwords(kind = "eng")


# Efeito de remover as stop words.
head(data$text, n = 1) %>%
  str_wrap(72) %>%
  cat("\n")

# Remoção das stop words.
data$text <- data$text %>%
  removeWords(words = c("bom", "muito", "pouco",
                        stopwords(kind = "pt")))

# Faz tokenização nas palavras individuais e empilha as palavras.
texto_un <- texto %>%
  unnest_tokens(output = "words", input = general)
texto_un

