#Carregando dados
df <- read.csv(
  file = "C:/MachineLearningR/MLDiabetes/diabetes.csv",
  header = TRUE,
  sep = ","
)
head(df)
str(df)

#Preparação dos dados

colSums(is.na(df))
table(df$Outcome)

df$Outcome <- as.factor(df$Outcome)
str(df)
#Verificando valores min, max, média, mediana...
summary(df$Insulin)

#Análisando alguns gráficos
boxplot(df)
boxplot(df$Insulin)

hist(df$Insulin)

# Análise exploratória
library(dplyr)
df2 <- df %>%
  filter(Insulin <= 250)
boxplot(df2$Insulin)

boxplot(df2$Pregnancies)
boxplot(df2$BMI)
boxplot(df2$Age)

summary(df2$Insulin)

# CONSTRUÇÃO DO MODELO
install.packages("caTools")
library(caTools)

# Divisão dos dados em treino e teste.
# 70% dos dados para treino e 30% dos dados para teste.

index <- sample.split(df2$Pregnancies, SplitRatio = .70)
index

train = subset(df2, index == TRUE)
test  = subset(df2, index == FALSE)

dim(df2)
dim(train)
dim(test)

install.packages("caret")
install.packages("e1071")
library(caret)
library(e1071)


#Treinando a primeira versão do modelo - KNN
modelo <- train(
  Outcome ~., data = train, method = "knn")

#Visualizando os resultados do modelo
modelo$results
modelo$bestTune


#Treinando a segunda versão do modelo - testando o comportamento do modelo com outros valores de k
modelo2 <- train (
  Outcome ~., data = train, method = "knn",
  tuneGrid = expand.grid(k = c(1:20)))

#Visualizando os resultados do modelo
modelo2$results
#Identificando o melhor valor de k
modelo2$bestTune

#Visualizando a performance do modelo - gráfico de linhas
plot(modelo2)


#Treinando a terceira versão do modelo - Naive bayes
install.packages("naivebayes")
library(naivebayes)


modelo3 <- train(
  Outcome ~., data = train, method = "naive_bayes")

#Visualizando os resultados do modelo
modelo3$results
modelo3$bestTune



#Treinando a quarta versão do modelo - randomForest
install.packages("randomForest")
library(randomForest)

modelo4 <- train(
  Outcome ~., data = train, method = "rpart2")

modelo4

#Verificando o peso e importância de cada variável 
varImp(modelo4$finalModel)

#As colunas "Insulin e Blood Pressure" não contribuem muito para o aprendizado do modelo

#Treinando o modelo sem as colunas "Insulin e BloodPressure" - train[,c(-3,-5)] exclui as colunas
modelo4_1 <- train(
  Outcome ~., data = train[,c(-3,-5)], method = "rpart2"
)
modelo4_1


# Visualizando a arvore de decisão
plot(modelo4_1$finalModel)
text(modelo4_1$finalModel)


#Modelo 5
install.packages("kernlab")
library(kernlab)

set.seed(100)
modelo5 <- train(
  Outcome ~., data = train, method = "svmRadialSigma"
  ,preProcess=c("center")
)

modelo5$results
modelo5$bestTune

#Avaliando o modelo
#Testando o modelo com os dados de teste
predicoes <- predict(modelo5,test)

predicoes


#Confunsion matrix para Verificar os resultados do modelo
confusionMatrix(predicoes, test$Outcome)


# Realizando predições

#Criando um dataframe apenas com o registro de um unico paciente para simular a utilização do modelo
novos.dados <- data.frame(
  Pregnancies = c(3),           
  Glucose = c(111.50),
  BloodPressure = c(70),
  SkinThickness = c(20),          
  Insulin = c(47.49),
  BMI = c(30.80),       
  DiabetesPedigreeFunction = c(0.34),
  Age = c(28)                     
)

novos.dados

#Utilizando o modelo para gerar a previsão - passando os dados do paciente
previsao <- predict(modelo5,novos.dados)
resultado <- ifelse(previsao == 1, "Positivo","Negativo")
#Verificando o resultado da predição do modelo
print(paste("Resultado:",resultado))


### VISUALIZAÇÃO DOS RESULTADOS

#Criando o arquivo com os resultados das predições
write.csv(predicoes,'resultado.csv')

#Lendo o arquivo de previsões que foi gerado
resultado.csv <- read.csv('resultado.csv')

#Alterando o nome das colunas do dataframe
names(resultado.csv) <- c('Indice','Valor previsto')

#Visualizando o dataframe
resultado.csv
