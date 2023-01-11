# Importing Datasets

install.packages('data.table')
install.packages('ranger')
install.packages('caret')
install.packages('caTools')
install.packages('pROC')
install.packages('rpart')
install.packages('rpart.plot')
install.packages('neuralnet')
install.packages("gbm")
install.packages("ROSE")
install.packages("smotefamily")

#...............................................................................


library(caTools)
library(ranger)
library(caret)
library(data.table)
library(pROC)
library(rpart)
library(rpart.plot)
library(neuralnet)
library(gbm, quietly=TRUE)


#...............................................................................


crdata <- read.csv("C:/Users/35846/Downloads/ml-latest-small/creditcard fraud detection/creditcard.csv")


str(crdata) # glance at the structure of the data set. last column class categorical column( other column numerical column)

dim(crdata) #rows and columns

head(crdata,6)

tail(crdata,6)

names(crdata) #column names

crdata$Class <- factor(crdata$Class, levels = c(0,1)) #Factor in R is also known as a categorical variable that stores both string and integer data values as levels

summary(crdata)

sum(is.na(crdata)) #null value checking

summary(crdata$Amount)


#Using $ Operator to Access Data Frame Column 
#table() function in R Language is used to create a categorical representation of data with variable name and the frequency in the form of a table.
#convert class to factor variable

#...............................................................................


d = table(crdata$Class) # get the distribution of fraud and legit transaction in dataset
 
p= prop.table(table(crdata$Class)) #percentage of the fraud and legit transaction


#pie chart of credit card transaction

labels <- c("legit", "fraud")
labels <- paste(labels, round(100*p, 2)) # paste for concatenate and c for combine,2 for to round the percentage for 2 digit
labels <- paste(labels, "%")

pie(d, labels, col = c("orange","red"),
    main = "Pie chart of credit card transaction") #title
#So this is a imbalanced dataset


#...............................................................................................


var(crdata$Amount) #var: Variance and standard deviation of complex vectors

sd(crdata$Amount)

head(crdata)

crdata$Amount=scale(crdata$Amount) #Scale() is a built-in R function that centers and/or scales the columns of a numeric matrix by default.

df = crdata[,-c(1)]

head(df)

#.........................................................................................................

#No model prediction

#consider or predict every single transaction is legitimate/ all the transaction is zero
library(caret)


predictions <- rep.int(0,nrow(crdata)) #repeat-rep and storing all the prediction here
predictions <- factor(predictions, levels = c(0,1))

confusionMatrix(data=predictions, reference = crdata$Class)

#..........................................................................................................

library(dplyr) # for smaller dataset

set.seed(1) #  to reproduce a particular sequence of 'random' numbers
crdata <- crdata %>% sample_frac(0.1) #if i only run this line will get random values everytime but with set.seed it will gime me same random value everytime


table(crdata$Class)

library(ggplot2)

ggplot(data = crdata, aes(x= V1, y = V2, col = Class))+ #2 claass 2 color
  geom_point()+  #scatter plot
  
  theme_bw() + #bw black nd white backgreound
  
  scale_color_manual(values = c('orange','red'))


#......................................................................................

# Creating train and test data

library(caTools)
set.seed(123)

data_sample = sample.split(crdata$Class,SplitRatio=0.80)

train_data = subset(crdata,sample = TRUE)

test_data = subset(crdata,data_sample==FALSE)

dim(train_data)

dim(test_data)
#.................................................................................

#Random Over-Sampling(ROS)

table(train_data$Class)

n_ligit <- 28437

new_frac_legit <- 0.50   #50% 0,1
new_n_total <- n_ligit/new_frac_legit #o make bigger number of fraud case in the dataset
                                       #more number of rows afteroversmpling



library(ROSE)

oversampling_result <- ovun.sample(Class ~ .,   #. seperated independeent variable
                                   data = train_data,
                                   method = "over", #check underand 
                                   N = new_n_total,
                                   seed = 2019)

oversampling_cr <-oversampling_result$data
table(oversampling_cr$Class)

ggplot(data = oversampling_cr, aes(x= V1, y = V2, col = Class))+ 
  geom_point(position =position_jitter(width = 0.2))+   #increase duplicate values
  theme_bw() + 
  scale_color_manual(values = c("dodgerblue", "red"))


#...............................................................................

#ROS and RUS

n_new <- nrow(train_data)
frac_fraud_new <- 0.50    #fraud and legitimate case same number

sampling_result <- ovun.sample(Class ~ .,   
                                   data = train_data,
                                   method = "both",
                                   N = n_new,
                                  p = frac_fraud_new,
                                   seed = 2019)

sampled_credit <- sampling_result$data
table(sampled_credit$Class)
prop.table(table(sampled_credit$Class))


ggplot(data = oversampling_cr, aes(x= V1, y = V2, col = Class))+ 
  geom_point(position =position_jitter(width = 0.2))+   #increase duplicate values
  theme_bw() + 
  scale_color_manual(values = c("dodgerblue", "red"))


#...............................................................................

#using smote to balance dataset

library(smotefamily)

table(train_data$Class)

#setting the number of fraud and legitamte cases and desired percentage of the legitimate cases

n0 <- 28437 #legi case
n1 <- 44   #fraud case
r0 <- 0.6  # after adding synthetic samples/ cases this will be rasio

#calculate the value for the dup_size parameter of smote

ntimes <- ((1-r0)/r0)*(n0/n1) -1

smote_result = SMOTE(X = train_data[,-c(1,31)],
                     target = train_data$Class,
                     K= 5,
                     dup_size = ntimes)

credit_smote <- smote_result$data #storing data

colnames(credit_smote[30]) <- "Class" #give the last/30th column name class

prop.table(table(credit_smote$Class))

#class ditribution for orginal dataset

ggplot(data = train_data, aes(x= V1, y = V2, color = Class))+ 
  geom_point()+
  theme_bw() + 
  scale_color_manual(values = c("dodgerblue", "red"))


#class ditribution for dataset using smote

ggplot(data = credit_smote, aes(x= V1, y = V2, color = Class))+ 
  geom_point()+
  scale_color_manual(values = c("dodgerblue", "red"))


#................................................................................

Logistic_Model=glm(Class~.,test_data,family=binomial())
summary(Logistic_Model)
# Visualizing summarized model through the following plots
plot(Logistic_Model)
# ROC Curve to assess the performance of the model

lr.predict <- predict(Logistic_Model,test_data, probability = TRUE)
auc.gbm = roc(test_data$Class, lr.predict, plot = TRUE, col = "blue")


#.................................................................................


# Fitting a Decision Tree Model

library(rpart)
library(rpart.plot)


decisionTree_model <- rpart(Class ~ . , crdata, method = 'class') #class dependent variable predictable

predicted_val <- predict(decisionTree_model, crdata, type = 'class')

probability <- predict(decisionTree_model, crdata, type = 'prob')

rpart.plot(decisionTree_model, extra = 0, type = 5, tweak = 1.2)

#...............................................................................

# Artificial Neural Network

library(neuralnet)

ANN_model =neuralnet (Class~.,train_data,linear.output=FALSE)

plot(ANN_model)

predANN=compute(ANN_model,test_data)

resultANN=predANN$net.result
resultANN=ifelse(resultANN>0.5,1,0)

#...............................................................................

# Gradient Boosting (GBM)

library(gbm, quietly=TRUE)
# Get the time to train the GBM model

system.time(
  model_gbm <- gbm(Class ~ .
                   , distribution = "bernoulli"
                   , data = rbind(train_data, test_data)
                   , n.trees = 500
                   , interaction.depth = 3
                   , n.minobsinnode = 100
                   , shrinkage = 0.01
                   , bag.fraction = 0.5
                   , train.fraction = nrow(train_data) / (nrow(train_data) + nrow(test_data))
  )
)
# Determine best iteration based on test data

gbm.iter = gbm.perf(model_gbm, method = "test")

model.influence = relative.influence(model_gbm, n.trees = gbm.iter, sort. = TRUE)


#Plot the gbm model
plot(model_gbm)


#...............................................................................

# Plot and calculate AUC on test data

library(pROC)
gbm_test = predict(model_gbm, newdata = test_data, n.trees = gbm.iter)
gbm_auc = roc(test_data$Class, gbm_test, plot = TRUE, col = "red")
print(gbm_auc)

