########################################
### Created by Valentino Gaffuri Bedetta
########################################
### on Jan 2019 @ Hult
### This is Fillmore
########################################
########################################


#### CLASSES OF R ####

############
# CLASS N1 #
############

"Hello World"

my_obj1<- "Hello World"
my_number1<-exp(1234)

############
# CLASS N2 #
############

num
summary<-summary(c(1,2,3,4,5,6,7,8,9,10))
print(summary)
sum<-sum(c(1,2,3,4,5,6,7,8,9,10))
product(c()) #is not working, find the correct function
quantile(c(1,2,3,4,5,6,7,8,9,10))
sd(c(1,2,3,4,5,6,7,8,9,10))
mean(c(1,2,3,4,5,6,7,8,9,10))

##### OBJECTS =====> 
# 1) SCALER, here we have some examples:
x<-2
y<-3
z<-exp(123)
my_obj<-"Hello World"

# 2) VECTORS
R_names<-c("Thomas","Eliz", "John")
R_seq<- seq(from=1,to=10,by=1)
print(R_seq)
R_erp<-rep(R_seq,each=3)
print(R_erp)
R_erp[6:10]

x<- c(1,2)
y<- c(3,4)
plot(x,y)

## 2 dimensions OBJECTS
# Matrix
my_matrix<-matrix(R_seq, nrow=2,ncol = 5, byrow=T) #here you can rd ncol and will work, R do the math
print(my_matrix)
my_matrix[,3:4] # this is to subset a matrix, before the coma you put the rows, and after the coma you put the columns

# DataFrames
colnames(german_credit_card_without_X)
german_credit_card_without_X$age
sum(german_credit_card_without_X$age)
sd(german_credit_card_without_X$age)
mean(german_credit_card_without_X$age)
summary(german_credit_card_without_X$age)
quantile(german_credit_card_without_X$age)

german_credit_card_without_X$age[3:9]
# when we take out one variable from the dataframe we get a VECTOR!
# A DataFrame and a Matrix is a bunch of vectors
# Remember that DF and Matrixes are 2 dimensional

# Here we create a subset from the SF, is different of just looking at a parto of the DF
# We need to create a vector name, but here we did not do it
german_credit_card_without_X[,c("duration","age")]
german_credit_card_without_X[which(german_credit_card_without_X$age<35),c("duration","age")]
german_credit_card_without_X[which(german_credit_card_without_X$age>35 & german_credit_card_without_X$age<50),c("duration","age")]
# here at hte end we applied some filters in the age column, but again did't create a vector, we need to do that to create another DF

new_subset<-subset(german_credit_card_without_X, age>35)
print(new_subset)


## Types of variables in R

is.numeric(german_credit_card_without_X$age) # the output here is a logical object (logical answer)
is.character(german_credit_card_without_X$age)

# Converting chategorical to numerical data
as.numeric(german_credit_card_without_X$good_bad)
gc<- as.data.frame(german_credit_card_without_X)
# here we will create a new variable "binary" and we will put good as 1
gc$binary<-gc$good_bad
gc[which(gc$binary=="good"), c("binary")] <- 1
print(gc$binary)

# here is a easier way to do it
gc$binary <-gsub("bad",0,gc$binary) #here we change with a simplet method the bas for 0s
print(gc$binary)

gc_clean<-na.omit(gc)
print(gc_clean)

############
# CLASS N3 #
############

library(readxl)
german_credit <- read_excel("Documents/Master USA - HULT/CLASSES/Module B/R/Data to work/german credit card.xls")
View(german_credit)

# To replace we can do it in 2 ways:
# One is less useless for business, because here yo can not put multiple condition
my_new_var <- gsub("X","",german_credit$purpose)
print(my_new_var)

# Here you can put "OR", "AND" to add more condition in the replacement
my_new_var1 <- replace(german_credit$purpose, which(german_credit$purpose=="X"),"")
print(my_new_var1)
# Here we use the "&" operators (remember that both X and Y need to be inside of the cell)
my_new_var2 <- replace(german_credit$purpose, which(german_credit$purpose=="X" & german_credit$purpose=="1"),"ex1")
print(my_new_var1)
# Here we use the "OR" and this is better, because will look for all the values but not in the same cell
my_new_var3 <- replace(german_credit$purpose, which(german_credit$purpose=="X" | german_credit$purpose=="1"),"ex1")
print(my_new_var1)
# Here you can replace when more than oe condition in diferent variables are meet
my_new_var4 <- replace(german_credit$age, which(german_credit$purpose=="X" & german_credit$duration>2),"100")
print(my_new_var1)

# Here we clean the data and we changed the good and bad for biniary,
# because is a character we need to use gsub, and then we transformed to numeric
german_credit$good_bad <- gsub("bad", "0", german_credit$good_bad)
german_credit$good_bad <- gsub("good", "1", german_credit$good_bad)
german_credit$good_bad <- as.numeric(german_credit$good_bad)

# SQL intro

library(rpart)
install.packages("sqldf")
# I have already install the package, so I don't need yo install it again
library(sqldf)

sql_german <- sqldf("
                    SELECT good_bad
                    FROM german_credit
                    WHERE history <4
                    ")


############
# CLASS N4 #
############

## SQL & Loops

# SQL
sql_german <- sqldf("
                    SELECT *
                    FROM german_credit
                    WHERE history <4
                    ")

#Here we are grouping
a <- c(1,1,0,0,1,1)
b <- c(25,30,25,30,40,40)
my_sql_df1 <- data.frame(V1=a,V2=b)
# here we can create a data frame, but to use the loops use the one below

# here we created en empty df with the matrix function
# we use this to use then loops, so we can fill in with date programaticaly
# is a good option to use the matrix to create a DF because you can create the dimension ->
# -> with the matrix function.
my_sql_df <- as.data.frame(matrix(nrow=6, ncol = 2))
my_sql_df$V1 <- a
my_sql_df$V2 <- b
# here we put the data inside of the matrix, you need to select the vectors where you ->
# -> want to put the data inside.

my_sql_df$V3 <- NULL
# here we delete the Vector that we dont use.

colnames(my_sql_df) <- c("good_bad", "age")
# here you are changing the namas of the variables inside the DF

colnames(my_sql_df)[1] <- c('binary_good')
# another way to change the mame of the variable (column), you can select the # of the variable.
colnames(my_sql_df[,1]) <- c('binary_good') # THIS IS WRONG! IS NOT THE WAY!

my_simple_sql <- sqldf("
                      SELECT age, SUM(binary_good)
                       FROM my_sql_df
                      GROUP BY age
                       ")
# here we use SQL to filter and SUM the data inside the DF

sql_german <- sqldf("
                    SELECT good_bad, COUNT(good_bad)
                    FROM german_credit
                    GROUP BY good_bad
                    ")
# here we COUNT the amount of each variable type.

sql_german_cond <- sqldf("
                         SELECT *, CASE WHEN good_bad=0 then 'bad'
                        WHEN good_bad=1 then 'good'
                        ELSE 'wrong' END AS new_var
                        FROM german_credit
                         ")
# here we create a new var in the DF with GOOD and BAD when there are 1 and 0.

## Programing = LOOPS
# FOR LOOP = 
#   1) create histograms
#   2) Create subsets
#   3) Replace values
#   4)

for (variable in vector) {
  
}
# structure of the FOR LOOP

my_vec <- c()
# here we create and empty vector to fill with the LOOP
for (i in 1:3) {
  my_vec[i] <-i
}# closing the i LOOP
# here we get the vector with the data inside

for (i in 1:nrow(german_credit)) {
  my_vec[i] <-german_credit$age[i] > 20
}
# here we do it on the column of the age i the DF of german_credit
# and also here we get FALSE or TRUE in all the columns

german_credit <- as.data.frame(german_credit)
my_var <- c()
for (i in 1:ncol(german_credit)) {
  try(my_var <- as.numeric(german_credit[,i]))
  try(hist(my_var))
  
}
# here we needed to create a DF, so that's why do the first german_credit <- as.data.frame(german_credit)
# Here be careful because there are some hist that are not being drawn because ->
# -> of the missing values inside of the variables.
# You need to do some cleaning before.

for (i in 1:ncol(german_credit)) {
  try(my_var <- german_credit[-which(is.na(german_credit[,i])),i])
  try(hist(my_var))
}
# this is not working but the professor told us that he will upload the result.

# HERE WE ARE GOING TO SEE CONDITIONAL STATEMENTS = IF STATEMENTS = results TRUE or FALSE
if(german_google$age[10]>20){print("Yay")}
if(german_google$age[10]<20){print("Yay")}
# here you have an easy example of how to use it (2 diff results)

# now you have with more conditions
if(german_google$age[10]>20){print("Yay")} else {"ble"}
if(german_google$age[10]<20){print("Yay")} else {"ble"}

# EXERCICE with the FOR LOOPS
for (i in 1:nrow(german_credit)) {
  if(german_credit$good_bad[i]==1){german_credit$good_bad[i] <- "good"}
  else{german_credit$good_bad[i]<-"bad"}
} # closing the i loop
# this is a really good way to replace the data


############
# CLASS N5 #
############

# Intrapolation - imputation
library(readr)
dataset_Facebook <- read_csv("Documents/Master USA - HULT/CLASSES/Module B/R/Data to work/dataset_Facebook.csv")
View(dataset_Facebook)

facebook<- dataset_Facebook

summary(facebook)

# the following code is a code that Tom uploaded in the annoucements in Kanvas
other_df <- facebook
for(i in 1:nrow(other_df)){
  if( is.na( other_df$like[i] ) ) { other_df$like[i] <- mean(other_df$like[-which(is.na(other_df$like))]) }
}

# here we do the mean of the original dateset, the code was taken from the previous script
# but we do the mean out of the whole script
facebook_mean <- mean(facebook$like[-which(is.na(facebook$like))])
facebook_mean

for(i in 1:nrow(facebook)){
  if( is.na( facebook$like[i] ) ) { facebook$like[i] <- mean(facebook$like[-which(is.na(facebook$like))]) }
}

# Here we can see that this 2 methods are the same, there's no difference.
# Because in the first one the ecuation will out the same mean as in the second one.
other_df$like - facebook$like
# that;s why we get 0 here, because is the same mean.

# we did some code in the file function_dataframe_conert... and we run a lopp there that is
# important


### Objetive achieve ###
## ALL THIS IS USE TO OPTIMIZE PORTFOLIO OF INVESMENT ##

# Here we instal this package to minimize (optimaze to get the min)
# here is to remain as close as possible to the line in the linear regression.
install.packages("minpack.lm")
library(minpack.lm)
y<-c(1,2,3,4)
x<-c(.5,.4,.7,.9)
x2<-c(.1,.2,.3,.4)

my_func <- function(b1, b2){
  y_est <- b1*x + b2*x2
  return(as.numeric(y_est))
}

# here we compare my real Y and the estimation = y_est
my_model <-nlsLM(y ~ my_func(b1,b2))
summary(my_model)


# Using this with an exercise that Tom gave
install.packages("minpack.lm")
library(minpack.lm)
y<-c(1000,480,1800,1000,990)
x<-c(1200,600,1050,860,720)
x2<-c(1050,310,2100,990,880)

my_func <- function(b1, b2){
  y_est <- b1*x + b2*x2
  return(as.numeric(y_est))
}

# here we compare my real Y and the estimation = y_est
my_model <-nlsLM(y ~ my_func(b1,b2))
summary(my_model)
#The weight is 31% for MIX 1 and 71% for mix2 = this is the result

# Here we adding one more mix to previuos one exercise
install.packages("minpack.lm")
library(minpack.lm)
y<-c(1000,480,1800,1000,990) #ojective to reach, as closest as possible
x<-c(1200,600,1050,860,720)
x2<-c(1050,310,2100,990,880)
x3<-c(710,420,1700,1600,1120)

my_func <- function(b1, b2,b3){
  y_est <- b1*x + b2*x2+b3*x3
  return(as.numeric(y_est))
}

# here we compare my real "Y" and the estimation = y_est
my_model <-nlsLM(y ~ my_func(b1,b2,b3))
summary(my_model)
# HERE WE GET AS A RESULT 25% OF MIX 1, 59% FOR MIX 2 AND 17% FOR MIX 3 #


# DICE GAME:
# WE CAN REPRODUCE THROUING TH DICE WITH A FUNCTION SAMPLE

my_uniform<-sample(1:6,1500, replace = T )
hist(my_uniform)
mean(my_uniform)

# Now let's do it with thw coins
my_binomial<-sample(1:2, 94, replace = T)
my_binomial
hist(my_binomial)
mean(my_binomial)

# now let's simulate the poison distribution
# the poition measure the amount of successes
?rpois
# the lamba is the mean here
hist(rpois(1000,2))
hist(rpois(1000,15))
# the bigger the mean the more normal distributed is the data



############################################
######## Exponential Distribution ##########
############################################

# how much time pass between succeses (from one sucess to other sucess)
# we can use this to see the depretiation of a machine or cars in companies if they are going down
# mean= 1/lamba ; variance= 1/lambda^2

#Examples:
# lambda is = to mean, so if we have an lambda of 10 we have a mean of 10 min also
# Let's build this with an histogram:
hist(rexp(1000, rate=10))#, breaks = seq(from=0, to=1000, by=100)) # here we do not use the breaks, so it automaticaly
# The distributions looks steep starting in 0 and going up to 0.8 in the X 

# Let's now put the example given in the PDF file Exponential distribution Cheat Sheet:
hist(rexp(1000, rate=.4), breaks = seq(from=0, to=30, by=1))

# Exercise in the PDF of exponential distribution Cheat Sheet:
# Let’s assume that we are talking about the time between each bus (line #1 in SF) at a given stop 
# (corner of California and Sansome). The X axis shows how many minutes elapsed since the previous bus. 
# In the chart above, 350 buses arrived sooner than 1 minute after the previous bus (blue circle). 
# There were also a few busses, maybe 10, that were 15 minutes apart (red circle). 
# You know that the distribution is exponential with a λ = 0.4
# The Muni CEO asked you to calculate the probability of the bus arriving:

# a) In more than 10 minutes after the previous bus.
# P(X>10)= exp(-λ*10)
exp(-.4*10)
# RESULT = 0.01831564 -> 1.8 % probability in more than 10 minutes (arrival)

# b) In less than 5 minutes after the previous bus.
# P(X<5) = 1 - exp(-λ*5)
1-exp(-.4*5)
# RESULT = 0.8646647 -> 86.47 % probability in less than 5 minutes (arrival)

############################################
############################################


############
# CLASS N6 #
############

my_means <- c()
for (i in 1:500){
  my_means [i] <-mean(sample(1:2,1000, replace = T))
}
hist(my_means, breaks = seq(from=1, to=2, by=.02))
# here the mean is going to 0.5 with more and more repetitions.
# this is the mean of the means of the samples.

## Some practicing in Exponential Distribution ##
hist(rexp(1000, rate = .2), breaks=seq(from=0, to=50,by=1))

## Exercise = > your company have 30 machines, .4 lambda, surivival exp distrib, 
# calculate, P more than 5 years, and, less than 1 year, and the same with a 1000 machines

# A)
exp(-.4*5)
# B)
1-exp(-.4*1)
# C) is the same probability


## Predicting Analytics
# logistical regression:
# we need to use binary if the variable is binomial

my_model<-glm(good_bad~age, data=german_credit, family = "binomial")
#exp(the result of the glm in the first column of the variable, the %)
summary(my_model)
# the result here was 0.0171
# the result is that the increase in the age will get you 1.0172 % of being good.
exp(0.1674)
# My result was different, don't know why
# But you can use this and put the exp to know how much increase or decrease


## Exercise with our dataset in the Census Income
my_model2<-glm(sex_binary~ hours_per_week, data = ci, family = "binomial")
summary(my_model2)
exp(0.043717)

####### IMPORTANT ########
# 1.044 = > every hour you increase the working hour, that increases the odds in 4.4% to being a male

# Function to make it easier
logit2prob <- function(x,coeff,intrcpt){
  logit <- intrcpt + x*coeff
  odds <- odds / (1+odds) # a one - unit change of x
  return(c(odds,prob))
}
logit2prob(x=20,coeff=0.017107,intrcpt = 0.242)
# this is not working but wil work if I correct the dataset German Credit with the 0 and 1 in good_bad

my_model3<-glm(good_bad~age+savings, data=german_credit, family = "binomial")
summary(my_model3)
exp(.015307)
exp(.275489)
#Estimate Std. Error z value Pr(>|z|)    
# (Intercept) -0.232612   0.247498  -0.940   0.3473    
# age          0.015307   0.006458   2.370   0.0178 *  
# savings      0.275489   0.051071   5.394 6.88e-08 ***


my_model4<-glm(sex_binary~ hours_per_week+age, data = ci, family = "binomial")
summary(my_model4)
exp(0.0429692)
exp(0.0117900)

# 1) how 1 increase in the variable impact the odds of success (of being good)
# 2) If I am 20 y what is the Probability of becoming good


### Data Visualization ###
# Use the page 147 for the exam to select the correct chart

library(ggplot2)
ggplot(data = german_credit, aes(age))+
  geom_histogram(binwidth = 5)

ggplot(data = german_credit, aes(age))+
  geom_dotplot()

ggplot(data = german_credit, aes(age, amount))+
         geom_point()

# let's go to the book to see how to apply the plot_ly
library(plotly)
p <- plot_ly(data=iris, x=~Sepal.Length, y=~Petal.Length, color= 'Species')
p


############################################
############################################


############
# CLASS N8 #
############

## Trees -> Prediction Model

# Here we use german credit with the good_bad (binary).
# Here we applied odd ratio (predictiing)
my_logit3<- glm(good_bad ~ age,data=german_credit, family="binomial")
summary(my_logit3)

# Here we take the following value to use the analysis:
# Coefficients:
#            Estimate Std. Error z value Pr(>|z|)   
#(Intercept) 0.215281   0.232343   0.927  0.35415   
#age    ###0.017886### 0.006413   2.789  0.00529 **
exp(0.017886)


# Here we add more variables in the analysis
my_logit3<- glm(good_bad ~ age+duration+coapp,data=german_credit, family="binomial")
summary(my_logit3)
exp(0.017842)
1-exp(-0.036954) # here you will have a decrease= the odd of succes go down 3.627
# Be carefull with the p-value, is huge (0.45148) so we need to eliminate
exp(0.116674)
#               Estimate Std. Error z value Pr(>|z|)    
#  (Intercept)  0.890704   0.326577   2.727  0.00638 ** 
#  age          0.017842   0.006620   2.695  0.00704 ** 
#  duration    -0.036954   0.005728  -6.452 1.11e-10 ***
#  coapp        0.116674   0.154954   0.753  0.45148



# Here we eliminate hte coapp because was a bad variable.
my_logit3<- glm(good_bad ~ age+duration,data=german_credit, family="binomial")
summary(my_logit3)
#             Estimate Std. Error z value Pr(>|z|)    
#(Intercept)  1.030311   0.269726   3.820 0.000134 ***
#  age          0.017703   0.006617   2.676 0.007460 ** 
#  duration    -0.037038   0.005725  -6.470  9.8e-11 ***

### Here we will start using TREES with TITANIC
# Here we will intall a few packages
# And run them after installing them
install.packages("rpart.plot")
install.packages("titanic")
install.packages("ROCR")
library(rpart)
library(titanic)
library(rpart.plot)
library(ROCR)

# first let's create a df with titanic
my_df_titanic <- as.data.frame(titanic_train)

# Now let's explore the data
summary(my_df_titanic)

# Now let's create a Logistic Regresion
my_titanic_champ <- glm(Survived ~ Pclass+Sex+Age+SibSp, data= my_df_titanic, family = "binomial")
summary(my_titanic_champ)
# here we see all the variables significant, less than 0

# here we take the values LN and transforme those to exp.
# We put the 1-exp() because the value is negative.
1-exp(-1.317398)
# Result = 73.21% if you change your class to one les class the probability of survive goes down 73.21%
1-exp(-2.623483)
# Result = 92.74% if you change your class to one les class the probability of survive goes down 92.74%
1-exp(-0.044385)
# Result = 4.34% if you change your class to one les class the probability of survive goes down 4.34%
1-exp(-0.376119)
# Result = 31.34% if you change your class to one les class the probability of survive goes down 31.34%

# Here we create the TREE
titanic_tree <- rpart(Survived ~ Pclass+Sex+Age+SibSp, data= my_df_titanic, method = "class")
rpart.plot(titanic_tree, type = 1, extra = 1)
# here we get the tree with the probabilities of the posibilities

# EXAM QUESTION ###################################
# If I want to get the model smaller.
# If the tree is too big is over fitting my data, I need to reduce the model fit
# If the tree is too small, too general, and I need to improve the model fit
# INCREASE THE CP AND YOU GET A smaller TREE, AND VICEVERSA:
# DECREASE THE CP VALUE - THE TREE gets bigger
titanic_tree <- rpart(Survived ~ Pclass+Sex+Age+SibSp, data= my_df_titanic, method = "class", cp=0.003)
rpart.plot(titanic_tree, type = 1, extra = 1)

# INCREASE THE CP VALUE - THE TREE gets SMALLER
titanic_tree <- rpart(Survived ~ Pclass+Sex+Age+SibSp, data= my_df_titanic, method = "class", cp=0.09)
rpart.plot(titanic_tree, type = 1, extra = 1)

# If we want to optimize the size we use the following function
# plotcp()
titanic_tree <- rpart(Survived ~ Pclass+Sex+Age+SibSp, data= my_df_titanic, method = "class", cp=0.01)
rpart.plot(titanic_tree, type = 1, extra = 1)
plotcp(titanic_tree)
# the better size of the cp value is the smallest in the graph.
# Here is 0.013

# we change the cp value
titanic_tree <- rpart(Survived ~ Pclass+Sex+Age+SibSp, data= my_df_titanic, method = "class", cp=0.013)
rpart.plot(titanic_tree, type = 1, extra = 1)
plotcp(titanic_tree)
# Now the better size is 0.015
# always try a few CP values to see what is the best for your purpose
# WE CAN COMPARE THE MODEL WITH A CHART (LIFT-GAINS CHART)
# WITH THESE CHART WE CAN SEE WHICH MODEL IS BETTER TO PRESENT AND TO SHOW
# AND SUGGEST WHICH IS THE BEST CHART.
# Pruning is getting the chart smaller -> and we need to generalize the model and
# --> not to improve the fit.


############################################
############################################


############
# CLASS N9 #
############

#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("titanic")
#install.packages("ROCR")
##############################
#Creating a logistic regression for titanic
##############################
library(rpart)
mydf_train <- as.data.frame(titanic_train)
summary(mydf_train)#let's see what's in the data
View(mydf_train)
#creating a champion model with the titanic_train dataset:
my_logistic <- glm(Survived ~ Pclass + Sex + Age+ SibSp, data = titanic_train, family="binomial")
summary(my_logistic)
#predicting probability of 1
predict_logit <- predict(my_logistic, mydf_train, type="response")#
print(predict_logit)
# here you get the probabilities of every passenger and the probability of living
# The probability of success


##############################
#Creating a tree for titanic
##############################
library(titanic)
mytree <- rpart(Survived ~ Pclass + Sex + Age+ SibSp, data = titanic_train, method = "class")#, control=rpart.control(minsplit=50, cp=0.013))
rpart.plot::rpart.plot(mytree, type = 1, extra=1, box.palette =c("pink", "green"), branch.lty=3, shadow.col = "gray")

plotcp(mytree)#getting the best cp value, do we need to go back and prune?
#use the cp value that has the lower error

############################################
#####The section below is optional##########
############################################
#The section below will explain the basics of
#model comparison and will code
#the lifts and gains chart
#### Create the lift and gains chart
#Scoring the model
library(ROCR)
mydf <- as.data.frame(titanic_train)
val_1 <- predict(mytree, mydf, type="prob")#We want to predict probability of 1 for each observations
print(val_1)
#Storing Model Performance Scores

pred_val <- prediction(val_1[,2], mydf$Survived) # tree regression
pred_val_logit <- prediction(predict_logit, mydf$Survived) # logistical regression
#we need performance
perf <- performance(pred_val,"tpr","fpr") # tree regression
perf_logit <- performance(pred_val_logit,"tpr","fpr") #logistical regression
#Plotting Lift Curve
plot(perf,col="black",lty=3, lwd=3)
plot(perf_logit,col="blue",lty=3, lwd=3, add=TRUE)
#plot(performance(pred_val, measure="lift", x.measure="rpp"), colorize=TRUE)


# Now let's use the prediction func to use in the german credit
# Here I copy again the model construction
my_logit3<- glm(good_bad ~ age+duration,data=german_credit, family="binomial")
summary(my_logit3)

# Here I construct the TREE model and make the graph
german_credit_tree <- rpart(good_bad ~ age+duration,data=german_credit, method = "class", cp=0.014)
rpart.plot(german_credit_tree, type = 1, extra = 1)
# here we can see that the main important split is with the duration and then age.

# let's do a plotcp to see if the size is OK
plotcp(german_credit_tree)
# now here we have the plots
rpart.plot::rpart.plot(mytree, type = 1, extra=1, box.palette =c("pink", "green"), branch.lty=3, shadow.col = "gray")
plotcp(mytree)


# here we try to put more variables into the tree, and let's see what happens
german_credit_tree_moreV <- rpart(good_bad ~ age+duration+savings+checking+coapp,data=german_credit, method = "class",cp=0.0065)
rpart.plot(german_credit_tree_moreV, type = 1, extra = 1)
plotcp(german_credit_tree_moreV)
# Here you can see that the model rid COAPP taht was not statisticaly signicant
# So here we can put all the variables and the model will rid the ones that 
# are not important.
# IN THE TREE INCLUDE MORE THAN YOU NEED BECAUSE THE MODEL WILL RID

# BUT REMEMBER TO COMPARE MODELS WITH THE SAME VARIABLES, ALWAYS!!!
# If you want to compare put the same variables in all the models

#creating a tree regression
german_credit_tree_moreV <- rpart(good_bad ~ age+duration+savings+checking+coapp,data=german_credit, method = "class",cp=0.0065)
rpart.plot(german_credit_tree_moreV, type = 1, extra = 1)
plotcp(german_credit_tree_moreV)

View(ci)
census_income_tree <- rpart(sex_binary ~ gnlwgt+hours_per_week+education_num+age,data=ci, method = "class",cp=0.0087)
rpart.plot(census_income_tree, type = 1, extra = 1)
plotcp(census_income_tree)


############################################
############################################


############
# CLASS N9 #
############

# Here we will see Shiny
# First we need to install the package

install.packages('rsconnect')
library('rsconnect')
# Here I went to create a new Shiny Web App in the buton in the left upper corner


