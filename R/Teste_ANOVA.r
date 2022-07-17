# Estatística na Prática 1 - Análise de Variância (ANOVA)

# Dados varejo
# Madeira A = 1 3 1 4 2 3 3 3 
# Madeira B = 6 8 5 3 8 9 3 4
# Madeira C = 6 8 6 6 7 5 5 5

# Lista
produto <- c(4,5,4,3,2,4,3,4,4,6,8,4,5,4,6,5,8,6,6,7,6,6,7,5,6,5,5)

# Lista dos produtos total de testes dos 3 produtos
produtos_geral <-c(rep("A", 2), rep("B", 7), rep("C", 6))
produtos_geral

# Criando um dataframe
df <- data.frame(produto, produtos_geral)
df

# Teste ANOVA
teste_anova <-  aov(produto ~ produtos_geral, data = df)
teste_anova

# Sumário do teste
summary(teste_anova)