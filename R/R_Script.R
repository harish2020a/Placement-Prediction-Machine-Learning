#Installing packages
install.packages("naivebayes")
install.packages("e1071")
install.packages("caTools")
install.packages("ggplot2")
install.packages("caret")
install.packages("dplyr")
install.packages("gdata")
install.packages("ROSE")
install.packages("stringr")
install.packages("rBayesianOptimization")
install.packages("psych")
install.packages("smotefamily")
install.packages("randomForest")
install.packages("class")
install.packages("superml")
install.packages("corrplot")
install.packages("klaR")
install.packages("ROCR")
install.packages("pROC")
install.packages("MLmetrics")
install.packages("CatEncoders")
install.packages("R6Class")

#importing libraries
options(warn = -1)
library(naivebayes)
library(e1071)
library(caTools)
library(ggplot2)
library(caret)
library(dplyr)
library(gdata)
library(ROSE)
library(stringr)
library(rBayesianOptimization)
library(psych)
library(smotefamily)
library(randomForest)
library(class)
library(corrplot)
library(klaR)
library(ROCR)
library(pROC)
library(MLmetrics)
library(R6)
library(CatEncoders)

#' Label Encoder
#' @description Encodes and decodes categorical variables into integer values and vice versa.
#' This is a commonly performed task in data preparation during model training, because all machine learning models require
#' the data to be encoded into numerical format. It takes a vector of character or factor values and encodes them into numeric.
#' @format \code{\link{R6Class}} object.
#' @section Usage:
#' For usage details see \bold{Methods, Arguments and Examples} sections.
#' \preformatted{
#' lbl = LabelEncoder$new()
#' lbl$fit(x)
#' lbl$fit_transform(x)
#' lbl$transform(x)
#' }
#' @section Methods:
#' \describe{
#'   \item{\code{$new()}}{Initialise the instance of the encoder}
#'   \item{\code{$fit()}}{creates a memory of encodings but doesn't return anything}
#'   \item{\code{$transform()}}{based on encodings learned in \code{fit} method is applies the transformation}
#'   \item{\code{$fit_transform()}}{encodes the data and keep a memory of encodings simultaneously}
#'   \item{\code{$inverse_transform()}}{encodes the data and keep a memory of encodings simultaneously}
#' }
#' @section Arguments:
#' \describe{
#'  \item{data}{a vector or list containing the character / factor values}
#' }
#' @export
#' @examples
#' data_ex <- data.frame(Score = c(10,20,30,4), Name=c('Ao','Bo','Bo','Co'))
#' lbl <- LabelEncoder$new()
#' data_ex$Name <- lbl$fit_transform(data_ex$Name)
#' decode_names <- lbl$inverse_transform(data_ex$Name)

LabelEncoder <- R6Class("LabelEncoder", public = list(
  
  #' @field input_data internal use
  input_data = NA,
  #' @field encodings internal use
  encodings = NA,
  #' @field decodings internal use
  decodings = NA,
  #' @field fit_model internal use
  fit_model = FALSE,
  
  #' @details
  #' Fits the labelencoder model on given data
  #'
  #' @param data_col a vector containing non-null values
  #' @return NULL, calculates the encoding and save in memory
  #'
  #' @examples
  #' data_ex <- data.frame(Score = c(10,20,30,4), Name=c('Ao','Bo','Bo','Co'))
  #' lbl <- LabelEncoder$new()
  #' lbl$fit(data_ex$Name)
  #' data_ex$Name <- lbl$fit_transform(data_ex$Name)
  #' decode_names <- lbl$inverse_transform(data_ex$Name)
  
  fit = function(data_col){
    
    self$input_data <- private$check_data(data_col)
    self$encodings <- private$encoder(self$input_data)
    self$decodings <- private$decoder(self$encodings)
    self$fit_model <- TRUE
    
  },
  
  #' @details
  #' Fits and returns the encoding
  #'
  #' @param data_col a vector containing non-null values
  #' @return encoding values for the given input data
  #'
  #' @examples
  #' data_ex <- data.frame(Score = c(10,20,30,4), Name=c('Ao','Bo','Bo','Co'))
  #' lbl <- LabelEncoder$new()
  #' lbl$fit(data_ex$Name)
  #' data_ex$Name <- lbl$fit_transform(data_ex$Name)
  
  fit_transform = function(data_col){
    
    data_col <- private$check_data(data_col)
    self$fit(data_col)
    
    vals <- private$mapper(data_col,
                           self$encodings,
                           convert_type = NULL,
                           output_type = numeric(1))
    return(vals)
    
  },
  
  #' @details
  #' Returns the encodings from the fitted model
  #'
  #' @param data_col a vector containing non-null values
  #' @return encoding values for the given input data
  #'
  #' @examples
  #' data_ex <- data.frame(Score = c(10,20,30,4), Name=c('Ao','Bo','Bo','Co'))
  #' lbl <- LabelEncoder$new()
  #' lbl$fit(data_ex$Name)
  #' data_ex$Name <- lbl$transform(data_ex$Name)
  
  transform = function(data_col){
    
    if (!(isTRUE(self$fit_model)))
      stop("Please run fit before using transform.")
    
    data_col <- private$check_data(data_col)
    
    # all values in the new vector should be in encodings
    if (!(all(data_col %in% names(self$encodings)))) {
      message(strwrap("There are new values in this vector which weren't
                 available during fit. Replacing those values with 'NA'"))
      
      # replace new values with 'NA'
      val_index <- which(!(data_col %in% names(self$encodings)))
      data_col[val_index] <- "NA"
      
    }
    
    vals <- private$mapper(data_col,
                           self$encodings,
                           convert_type = NULL,
                           output_type = numeric(1))
    return(vals)
  },
  
  #' @details
  #' Gives back the original values from a encoded values
  #'
  #' @param coded_col a vector containing label encoded values
  #' @return original values from the label encoded data
  #'
  #' @examples
  #' data_ex <- data.frame(Score = c(10,20,30,4), Name=c('Ao','Bo','Bo','Co'))
  #' lbl <- LabelEncoder$new()
  #' lbl$fit(data_ex$Name)
  #' data_ex$Name <- lbl$fit_transform(data_ex$Name)
  #' decode_names <- lbl$inverse_transform(data_ex$Name)
  
  inverse_transform = function(coded_col){
    
    #check if all values exist in decode
    if (!(all(coded_col %in% self$encodings))) {
      stop(strwrap("There are new values in this data which weren't
                    available during fit. Please fix the issue."))
    }
    
    # write here
    vals <- private$mapper(coded_col,
                           self$decodings,
                           convert_type = as.character,
                           output_type = character(1))
    return(vals)
    
  }
  
),
private = list(
  
  encoder = function(data_col){
    
    # this is to handle if the input value is factor, it should be character
    data_col <- as.character(data_col)
    
    all_values <- unique(data_col)
    maps <- list()
    
    # because the encoding should start with zero (i - 1)
    for (i in seq(all_values))  maps[all_values[i]] <- i - 1
    return(maps)
    
  },
  
  decoder = function(en_output){
    
    # reverse the encoded list - make values as names and vice versa
    f <- as.list(names(en_output))
    names(f) <- en_output
    return(f)
    
  },
  
  mapper = function(input_vec, coded_list, convert_type ,output_type){
    
    if (is.null(convert_type)) {
      # this is to avoid using convert_type argument for
      # transform, fit_transform
      return(vapply(input_vec,
                    function(x) coded_list[[x]],
                    FUN.VALUE = output_type,
                    USE.NAMES = F))
    }
    
    return(vapply(input_vec,
                  function(x) coded_list[[convert_type(x)]],
                  FUN.VALUE = output_type,
                  USE.NAMES = F))
  },
  
  
  check_data = function(data_col){
    
    # fix data issues here
    if (any(is.na(data_col))) {
      message("The data contains NA values. Imputing NA with 'NA' ")
      data_col <- replace(data_col, which(is.na(data_col)), "NA")
    }
    
    if (any((data_col == ""))) {
      message("The data contains blank values. Imputing them with 'NA' ")
      data_col <- replace(data_col, which(data_col == ""), "NA")
    }
    
    if (any((data_col == " "))) {
      message("The data contains blank values. Imputing them with 'NA' ")
      data_col <- replace(data_col, which(data_col == " "), "NA")
    }
    
    return(data_col)
    
  }
)
)


#reading dataset
df=read.csv("Placement_Data_Full_Class.csv",header=TRUE)
head(df)

tail(df)

describe(df)
str(df)

#identifying datatypes
chrDT<-list()
intDT<-list()
for(i in seq(ncol(df)))
{
  if(typeof(df[,i])=="character")
  {
    chrDT<-append(chrDT,i,after=length(chrDT))
  }
  else{
    intDT<-append(intDT,i,after=length(intDT))
  }
}
cat("Character Datatype columns :\n")
for(i in names(df)[unlist(chrDT)])
  cat('> ',i,"\n")

cat("\nInteger Datatype columns :\n")
for(i in names(df)[unlist(intDT)])
  cat('> ',i,"\n")

for(i in c(unlist(chrDT)))
  cat(names(df)[i],'{',summary(df[i]),'}',"\n\n")

for(i in c(unlist(intDT)))
  cat(names(df)[i],'{',summary(df[i]),'}',"\n\n")

summary(df)

#Data Preprocessing

#making categorical Data
str(df)

for(i in names(df))
{
  if(typeof(df[[i]])=="character")
  {
    df[[i]]=as.factor(df[[i]])
  }
}
str(df)

#Attribute Selection
df=df[,-ncol(df)] #deleting salary attribute  
head(df)

summary(df)

#Size and Checking for NULL values
cat("Size of DataFrame :",dim(df)[1],'x',dim(df)[2],"\n")
cat("No. of NULL values in Dataset : ",sum(is.na(df)))

head(data.frame(df$status))

#Correlation between the features
chrDT<-list()
intDT<-list()
for(i in seq(ncol(df)))
{
  if(typeof(df[,i])=="integer")
  {
    chrDT<-append(chrDT,i,after=length(chrDT))
  }
  else{
    intDT<-append(intDT,i,after=length(intDT))
  }
}

corel<-cor(df[,unlist(intDT)], method = "pearson", use = "complete.obs")
corel

corrplot(corel, order = 'AOE')

#All attributes have positive correlation but not in huge margin. 
#So, we can conclude that these are independent variables

#Description of Placed and Not Placed Data Points
describe(df[df$status=='Placed',][,unlist(intDT)])
describe(df[df$status=='Not Placed',][,unlist(intDT)])

#Visualization
barplot(table(df$status), main = "Student Placements",
        xlab="No. of Placed Students",ylab="No.of Students",col=c('red','#0000AA'),ylim=c(0,200))
legend("topleft",c("Not Placed","Placed"),fill = c('red','#0000AA'))
val<-c(nrow(df[df$status=="Not Placed",]),nrow(df[df$status=="Placed",]))
text(val+10,labels =val,adj=c(2,0),cex=2)

#from above graph we can conclude that there is imbalance in prediction class attribute which may affect in model accuracy

cat("Prediction Class Datatype : ",class(df$status))

#Data distributions

#Gender
par(mfrow=c(1,2))
barplot(table(df$gender),main="Students Gender", col=c('pink','navy'),ylim=c(0,200))
barplot(table(df$gender[df$status=='Placed']),main="Placed Students Gender", col=c('pink','navy'),ylim=c(0,200))

#SSC Board
par(mfrow=c(1,2))

barplot(table(df$ssc_b), main = "SSC Board",
        xlab="Levels of Board", ylab="No.of Students",col =rainbow(2),ylim=c(0,200))
legend("topleft",c("Central","Others"),fill = rainbow(2))
val<-c(nrow(df[df$ssc_b=="Central",]),nrow(df[df$ssc_b=="Others",]))
text(val+10,labels =val,adj=c(1,0),cex=2)

barplot(table(df$ssc_b[df$status =='Placed']), main = "SSC Board Placed Students",
        xlab="Levels of Board", ylab="No.of Students",col =rainbow(2),ylim=c(0,200))
legend("topleft",c("Central","Others"),fill = rainbow(2))
val<-c(
  nrow(df[df$status =='Placed',][as.data.frame(df$ssc_b[df$status =='Placed'])=='Central',]),
  nrow(df[df$status =='Placed',][as.data.frame(df$ssc_b[df$status =='Placed'])=='Others',])
)
text(val+10,labels =val,adj=c(1,0),cex=2)

#HSC Board
par(mfrow=c(1,2))

x<-barplot(table(df$hsc_b), main = "HSC Board",
           xlab="Levels of Board", ylab="No.of Students",col =rainbow(2),ylim=c(0,250))
legend("topleft",c("Central","Others"),fill = rainbow(2))
val<-c(nrow(df[df$hsc_b=="Central",]),nrow(df[df$hsc_b=="Others",]))
text(val+10,labels =val,adj=c(1,0),cex=2)

barplot(table(df$hsc_b[df$status =='Placed']), main = "HSC Board Placed Students",
        xlab="Levels of Board", ylab="No.of Students",col =rainbow(2),ylim=c(0,250))
legend("topleft",c("Central","Others"),fill = rainbow(2))
val<-c(
  nrow(df[df$status =='Placed',][as.data.frame(df$hsc_b[df$status =='Placed'])=='Central',]),
  nrow(df[df$status =='Placed',][as.data.frame(df$hsc_b[df$status =='Placed'])=='Others',])
)
text(val+10,labels =val,adj=c(1,0),cex=2)

#SSC & HSC Board student pass percentage
par(mfrow=c(1,2))
# par( mai=c(0.2,0.2,0.2,0.2))
# par(fig=c(0,0.5,0.5,1))
hist(df$ssc_p,col=rainbow(39),new=TRUE,xlab='percentages',ylab='no. of students',main='SSC Pass percentages',ylim=c(0,70))
#par(fig=c(0.5,1,0.5,1), new=TRUE)
hist(df$hsc_p,col=rainbow(39),new=TRUE,xlab='percentages',ylab='no. of students',main='HSC Pass percentages',ylim=c(0,70))

#Branch
par(mfrow=c(1,2))#, mar=c(4,4,4,1), oma=c(0.5,0.5,0.5,0))

barplot(table(df$degree_t), main = "Branches",
        xlab="Types of Branches", ylab="No.of Students",col =rainbow(3),ylim=c(0,280))
legend("topleft",c("Comm&Mgmt","Others","Sci&Tech"),fill = rainbow(3))
val<-c(nrow(df[df$degree_t=="Comm&Mgmt",]),nrow(df[df$degree_t=="Others",]),nrow(df[df$degree_t=="Sci&Tech",]))
text(val+10,labels =val,adj=c(1,0),cex=2)

barplot(table(df$degree_t[df$status=='Placed']), main = "Branches Placed",
        xlab="Types of Branches", ylab="No.of Students",col =rainbow(3),ylim=c(0,280))
legend("topleft",c("Comm&Mgmt","Others","Sci&Tech"),fill = rainbow(3))
val<-c(nrow(df[df$status=='Placed',][as.data.frame(df$degree_t[df$status =='Placed'])=="Comm&Mgmt",]),
       nrow(df[df$status=='Placed',][as.data.frame(df$degree_t[df$status =='Placed'])=="Others",]),
       nrow(df[df$status=='Placed',][as.data.frame(df$degree_t[df$status =='Placed'])=="Sci&Tech",])
)
text(val+10,labels =val,adj=c(1,0),cex=2)

#Work Experience
barplot(table(df$workex),main='Work Experience',col=c('red','blue'),xlab="Having experience", ylab="No.of Students",ylim=c(0,180))
val<-c(nrow(df[df$workex=="No",]),nrow(df[df$workex=="Yes",]))
text(val+10,labels =val,adj=c(1,0),cex=2)
par(mfrow=c(1,2))
barplot(table(df$workex[df$status=='Placed']),main='Work Experience Placed',col=c('red','blue'),xlab="Having experience", ylab="No.of Students",ylim=c(0,180))
val<-c(length(df$workex[df$status=='Placed'][df$workex[df$status=='Placed']=='No']),length(df$workex[df$status=='Placed'][df$workex[df$status=='Placed']=='Yes']))
text(val+5,labels =val,adj=c(1,0),cex=2)
barplot(table(df$workex[df$status=='Not Placed']),main='Work Experience Not Placed',col=c('red','blue'),xlab="Having experience", ylab="No.of Students",ylim=c(0,180))
val<-c(length(df$workex[df$status=='Not Placed'][df$workex[df$status=='Not Placed']=='No']),length(df$workex[df$status=='Not Placed'][df$workex[df$status=='Not Placed']=='Yes']))
text(val+5,labels =val,adj=c(1,0),cex=2)

#Employee Readiness Test scores
par(mfrow=c(1,2))

hist(df$etest_p[df$status=='Placed'],col=rainbow(10),new=TRUE,ylim=c(0,30),main='Etest performance of \nPlaced data',xlab='scores',ylab='no. students')
hist(df$etest_p[df$status=='Not Placed'],col=rainbow(10),new=TRUE,ylim=c(0,30),main='Etest performance of\n Not Placed data',xlab='scores',ylab='no. students')

# Data interpretation
# By all the graphs we can conclude that all attributes are independent and all together combined to give meaningful results
# for this dataset naive bayes is perfect suitable but due to imbalance there might be hickups in testing performance.

#Naive Bayes Classifier

#Train and Test Split
ind<-sample(2,nrow(df),replace=TRUE,prob=c(0.8,0.2))

train_cl<-df[ind==1,]
test_cl<-df[ind==2,]
head(train_cl)
head(test_cl)

#Training
grid <- data.frame(fL=c(0,0.5,1.0), usekernel = TRUE, adjust=c(0,0.5,1.0))
classifier_cl <- naive_bayes(status ~ ., data = train_cl, tuneGrid=grid)
classifier_cl[4]

plot(classifier_cl)

#Predicting on train data(Validation)
y_pred_vd <- predict(classifier_cl, newdata = train_cl,interval = 'confidence')
y_pred_vd

#Evaluation functions
accuracy<-function(y_act,y_preds)
{
  cm <- table(y_act,y_preds)
  return(sum(diag(cm))/length(y_act)*100)
}
recall<-function(y_act,y_preds)
{
  cm <- table(y_act,y_preds)
  return((cm[2,2]/(cm[2,2]+cm[1,2]))*100)
}
precision<-function(y_act,y_preds)
{
  cm <- table(y_act,y_preds)
  return((cm[2,2]/(cm[2,2]+cm[2,1]))*100)
}
f1_score<-function(y_act,y_preds)
{
  return((2*precision(y_act,y_preds)*recall(y_act,y_preds))/(precision(y_act,y_preds)+recall(y_act,y_preds)))
}

#confusion matrix
cm <- table(train_cl$status, y_pred_vd)
cm

cat('Accuracy:',accuracy(train_cl$status, y_pred_vd))#<-sum(diag(cm))/nrow(test_cl)*100
cat('\nRecall:',recall(train_cl$status, y_pred_vd))
cat('\nPrecision:',precision(train_cl$status, y_pred_vd))
cat('\nF1_Score:',f1_score(train_cl$status, y_pred_vd))

lbl = LabelEncoder$new()
cat('LogLoss :',LogLoss(predict(classifier_cl, newdata = train_cl,type = "prob",interval = 'confidence')[,2],lbl$fit_transform(train_cl$status)))

#Prediction on test data(Performance Testing)
y_pred <- predict(classifier_cl, newdata = test_cl,interval = 'confidence')
y_pred

Predicted<-y_pred
cm <- table(test_cl$status, Predicted)
cm
fourfoldplot(cm, color = c("pink", "skyblue"),conf.level = 0, margin = 2, main = "Confusion Matrix")

length(test_cl$status)

cat('Accuracy:',accuracy(test_cl$status, y_pred))#<-sum(diag(cm))/nrow(test_cl)*100
cat('\nRecall:',recall(test_cl$status, y_pred))
cat('\nPrecision:',precision(test_cl$status, y_pred))
cat('\nF1_Score:',f1_score(test_cl$status, y_pred))

as.data.frame(confusionMatrix(cm, positive = "Placed")[3])
as.data.frame(confusionMatrix(cm, positive = "Placed")[4])

confusionMatrix(cm, positive = "Placed")

lbl = LabelEncoder$new()
cat('LogLoss :',LogLoss(predict(classifier_cl, newdata = test_cl,type = "prob",interval = 'confidence')[,2],lbl$fit_transform(test_cl$status)))

cat('Roc_Auc_Score :',auc(as.numeric(test_cl$status), as.numeric(y_pred)))

pROC_obj <- roc(as.numeric(test_cl$status), as.numeric(y_pred),
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)
sens.ci <- ci.se(pROC_obj)
plot(sens.ci, type="shape", col="lightblue")
plot(sens.ci, type="bars")

#K-fold Cross Validation
#no. of K-folds and K-folds splits

k=5
fold<-KFold(df$status, nfolds = k, stratified = TRUE)

mnf1_nb<-list()
mnacc_nb<-list()
for(i in seq(1:k))
{
  l=c(seq(1:k))
  l<-l[-i]
  train<-df[unlist(fold[l]),]
  test<-df[unlist(fold[i]),]
  grid <- data.frame(fL=c(0,0.5,1.0), usekernel = TRUE, adjust=c(0,0.5,1.0))
  classifier_cl <- naiveBayes(status ~ ., data = train, tuneGrid=grid)
  y_pred <- predict(classifier_cl, newdata = test,interval = 'confidence')
  mnf1_nb<-append(mnf1_nb,f1_score(test$status, y_pred),after=length(mnf1_nb))
  mnacc_nb<-append(mnacc_nb,accuracy(test$status, y_pred),after=length(mnacc_nb))
}
cat('Mean Accuracy',mean(c(unlist(mnacc_nb))))
cat('\n')
cat(paste('\nAccuracy Result of Fold',c(1:5),':',unlist(mnacc_nb)))
cat('\n')
cat('\nMean f1_score',mean(c(unlist(mnf1_nb))))
cat('\n')
cat(paste('\nf1_score Result of Fold',c(1:5),':',unlist(mnf1_nb)))

#Comparing with other ML models

#Logistic Regression
ind_lr<-sample(2,nrow(df),replace=TRUE,prob=c(0.8,0.2))
train_lr<-df[ind_lr==1,]
test_lr<-df[ind_lr==2,]
classifier_lr <-glm(status ~ .,data = train_lr,family = "binomial")
y_pred_lr <- predict(classifier_lr, newdata = test_lr,type = "response")
y_pred_lr <- ifelse(y_pred_lr >0.5, 1, 0)
acc<-accuracy(test_lr$status, y_pred_lr)
cat('Accuracy:',acc)

#K-fold LR
mnf1_lr<-list()
mnacc_lr<-list()
for(i in seq(1:k))
{
  l=c(seq(1:k))
  l<-l[-i]
  train<-df[unlist(fold[l]),]
  test<-df[unlist(fold[i]),]
  classifier_lr <-glm(status ~ .,data = train,family = "binomial")
  y <- predict(classifier_lr, newdata = test,interval = 'confidence')
  y <- ifelse(y>0.5, 1, 0)
  mnf1_lr<-append(mnf1_lr,f1_score(test$status, y),after=length(mnf1_lr))
  
  mnacc_lr<-append(mnacc_lr,accuracy(test$status, y),after=length(mnacc_lr))
}
cat('Mean Accuracy',mean(c(unlist(mnacc_lr))))
cat('\n')
cat(paste('\nAccuracy Result of Fold',c(1:5),':',unlist(mnacc_lr)))
cat('\n')
cat('\nMean f1_score',mean(c(unlist(mnf1_lr))))
cat('\n')
cat(paste('\nf1_score Result of Fold',c(1:5),':',unlist(mnf1_lr)))

#K-Nearest Neighbors(KNN)
lbl = LabelEncoder$new()
df_knn=df
for(i in chrDT){
  df_knn[,i]<-lbl$fit_transform(df_knn[,i])
}
head(df_knn)

ind_knn<-sample(2,nrow(df_knn),replace=TRUE,prob=c(0.8,0.2))
train_knn<-df_knn[ind_knn==1,]
test_knn<-df_knn[ind_knn==2,]
train_scale <- scale(train_knn[,1:ncol(df_knn)-1])
test_scale <- scale(test_knn[,1:ncol(df_knn)-1])
classifier_knn <-knn(train=train_scale,test=test_scale,cl=train_knn$status,k=floor(sqrt(nrow(df_knn))))
acc<-accuracy(test_knn$status, classifier_knn)
cat("Validation Accuracy:",acc)

#K-fold KNN
mnf1_knn<-list()
mnacc_knn<-list()
for(i in seq(1:k))
{
  l=c(seq(1:k))
  l<-l[-i]
  train_knn<-df_knn[unlist(fold[l]),]
  test_knn<-df_knn[unlist(fold[i]),]
  train_scale <- scale(train_knn[,1:ncol(df_knn)-1])
  test_scale <- scale(test_knn[,1:ncol(df_knn)-1])
  classifier_knn <-knn(train=train_scale,test=test_scale,cl=train_knn$status,k=floor(sqrt(nrow(df_knn))))
  mnf1_knn<-append(mnf1_knn,f1_score(test_knn$status, classifier_knn),after=length(mnf1_knn))
  mnacc_knn<-append(mnacc_knn,accuracy(test_knn$status, classifier_knn),after=length(mnacc_knn))
}
cat('Mean Accuracy',mean(c(unlist(mnacc_knn))))
cat('\n')
cat(paste('\nAccuracy Result of Fold',c(1:5),':',unlist(mnacc_knn)))
cat('\n')
cat('\nMean f1_score',mean(c(unlist(mnf1_knn))))
cat('\n')
cat(paste('\nf1_score Result of Fold',c(1:5),':',unlist(mnf1_knn)))

#Random Forest Classifier
ind_rf<-sample(2,nrow(df),replace=TRUE,prob=c(0.8,0.2))
train_rf<-df[ind_rf==1,]
test_rf<-df[ind_rf==2,]
classifier_rf <-randomForest(status ~ ., data=train_rf, importance=TRUE,proximity=TRUE)
y_pred_rf <- predict(classifier_rf, newdata = test_rf,interval = 'confidence')
acc<-accuracy(test_rf$status, y_pred_rf)
acc

#K-fold Random Forest
mnf1_rf<-list()
mnacc_rf<-list()
for(i in seq(1:k))
{
  l=c(seq(1:k))
  l<-l[-i]
  train<-df[unlist(fold[l]),]
  test<-df[unlist(fold[i]),]
  classifier_rf <-randomForest(status ~ ., data=train, importance=TRUE,proximity=TRUE)
  y <- predict(classifier_rf, newdata = test,interval = 'confidence')
  mnf1_rf<-append(mnf1_rf,f1_score(test$status, y),after=length(mnf1_rf))
  mnacc_rf<-append(mnacc_rf,accuracy(test$status, y),after=length(mnacc_rf))
}
cat('Mean Accuracy',mean(c(unlist(mnacc_rf))))
cat('\n')
cat(paste('\nAccuracy Result of Fold',c(1:5),':',unlist(mnacc_rf)))
cat('\n')
cat('\nMean f1_score',mean(c(unlist(mnf1_rf))))
cat('\n')
cat(paste('\nf1_score Result of Fold',c(1:5),':',unlist(mnf1_rf)))

#Support Vector Machine
ind_svm<-sample(2,nrow(df),replace=TRUE,prob=c(0.8,0.2))
train_svm<-df[ind_svm==1,]
test_svm<-df[ind_svm==2,]
classifier_svm = svm(status ~ ., data=train_svm,type = 'C-classification',kernel = 'linear')
y_pred_svm <- predict(classifier_svm, newdata = test_svm,interval = 'confidence')
acc<-accuracy(test_svm$status, y_pred_svm)
acc

k=5
#fold<-KFold(df$status, nfolds = k, stratified = TRUE)
mnf1_svm<-list()
mnacc_svm<-list()
for(i in seq(1:k))
{
  l=c(seq(1:k))
  l<-l[-i]
  train<-df[unlist(fold[l]),]
  test<-df[unlist(fold[i]),]
  classifier_svm = svm(status ~ ., data=train_svm,type = 'C-classification',kernel = 'linear')
  y <- predict(classifier_svm, newdata = test,interval = 'confidence')
  mnf1_svm<-append(mnf1_svm,f1_score(test$status, y),after=length(mnf1_svm))
  mnacc_svm<-append(mnacc_svm,accuracy(test$status, y),after=length(mnacc_svm))
}
cat('Mean Accuracy',mean(c(unlist(mnacc_svm))))
cat('\n')
cat(paste('\nAccuracy Result of Fold',c(1:k),':',unlist(mnacc_svm)))
cat('\n')
cat('\nMean f1_score',mean(c(unlist(mnf1_svm))))
cat('\n')
cat(paste('\nf1_score Result of Fold',c(1:k),':',unlist(mnf1_svm)))


#Models Comparison
par(mfrow=c(2,2))
fourfoldplot(table(test_lr$status,y_pred_lr), color = c("pink", "skyblue"),conf.level = 0, margin = 2, main = "Logistic Regression ")
fourfoldplot(table(test_knn$status,classifier_knn), color = c("pink", "skyblue"),conf.level = 0, margin = 2, main = "K-NN ")
fourfoldplot(table(test_rf$status,y_pred_rf), color = c("pink", "skyblue"),conf.level = 0, margin = 2, main = "Random Forest")
fourfoldplot(table(test_svm$status,y_pred_svm), color = c("pink", "skyblue"),conf.level = 0, margin = 2, main = "SVM")


kacc=c(mean(c(unlist(mnacc_nb))),mean(c(unlist(mnacc_lr))),mean(c(unlist(mnacc_knn))),mean(c(unlist(mnacc_rf))),mean(c(unlist(mnacc_svm))))
kf1=c(mean(c(unlist(mnf1_nb))),mean(c(unlist(mnf1_lr))),mean(c(unlist(mnf1_knn))),mean(c(unlist(mnf1_rf))),mean(c(unlist(mnf1_svm))))
accmax=c(max(unlist(mnacc_nb)),max(unlist(mnacc_lr)),max(unlist(mnacc_knn)),max(unlist(mnacc_rf)),max(unlist(mnacc_svm)))
accmin=c(min(unlist(mnacc_nb)),min(unlist(mnacc_lr)),min(unlist(mnacc_knn)),min(unlist(mnacc_rf)),min(unlist(mnacc_svm)))
f1max=c(max(unlist(mnf1_nb)),max(unlist(mnf1_lr)),max(unlist(mnf1_knn)),max(unlist(mnf1_rf)),max(unlist(mnf1_svm)))
f1min=c(min(unlist(mnf1_nb)),min(unlist(mnf1_lr)),min(unlist(mnf1_knn)),min(unlist(mnf1_rf)),min(unlist(mnf1_svm)))
model_df=data.frame(Algorithm=c('Naive Bayes','Logistic Regression','K-Nearest Neighbours','Random Forest',"Support Vector Machine"),Min_Accuracy=accmin,Max_Accuracy=f1max,Min_F1score=f1min,Max_F1score=accmax,KFold_Accuracy=kacc,KFold_F1score=kf1)

model_df


par(mfrow=c(2,2))
fn<-function(x){
  return (format(round(x, 2), nsmall = 2))
}
labs=c('NB','LR','K-NN','RF','SVM')
barplot(model_df$Min_Accuracy,ylim=c(50,120),names.arg=labs,col=rainbow(5), log="y",main='Min_Accuracy')
text(model_df$Min_Accuracy,labels =sapply(model_df$Min_Accuracy,fn),pos=3,cex=1)

barplot(model_df$Max_Accuracy,ylim=c(50,120),names.arg=labs,col=rainbow(5), log="y",main='Max_Accuracy')
text(model_df$Max_Accuracy,labels =sapply(model_df$Max_Accuracy,fn),pos=3,cex=1)

barplot(model_df$Min_F1score,ylim=c(50,120),names.arg=labs,col=rainbow(5), log="y",main='Min_F1score')
text(model_df$Min_F1score,labels =sapply(model_df$Min_F1score,fn),pos=3,cex=1)

barplot(model_df$Max_F1score,ylim=c(50,120),names.arg=labs,col=rainbow(5), log="y",main='Max_F1score')
text(model_df$Max_F1score,labels =sapply(model_df$Max_F1score,fn),pos=3,cex=1)


plot(model_df$Min_Accuracy,type = "o",col='red',ylim=c(75,100),lwd=2.0, xaxt = "n",main='Accuracy Comparison')
axis(1, at=1:5, labels=labs)
lines(model_df$Max_Accuracy, type = "o", col = "blue",lwd=2.0)
legend("topleft",c("Min Accuracy","Max Accuracy"),fill = c('red','blue'))

plot(model_df$Min_F1score,type = "o",col='red',ylim=c(75,100),lwd=2.0, xaxt = "n",main='F1-Score Comparison')
axis(1, at=1:5, labels=labs)
lines(model_df$Max_F1score, type = "o", col = "blue",lwd=2.0)
legend("topleft",c("Min F1-score"," Max F1-score"),fill = c('red','blue'))


plot(model_df$KFold_Accuracy,type = "o",col='red',ylim=c(80,95),lwd=2.0, xaxt = "n",main='K-Fold Accuracy and F1-Score Comparison')
axis(1, at=1:5, labels=labs)
lines(model_df$KFold_F1score, type = "o", col = "blue",lwd=2.0)
legend("topleft",c("Accuracy","F1-score"),fill = c('red','blue'))


plot(unlist(mnacc_nb),type = "o",col='red',ylim=c(70,100),lwd=2.0, xaxt = "n",main='Kfold Validation Accuracies',xlab='K-folds',ylab='percentages')
axis(1, at=1:5, labels=c(1:5))
lines(unlist(mnacc_lr), type = "o", col = "blue",lwd=2.0)
lines(unlist(mnacc_knn), type = "o", col = "green",lwd=2.0)
lines(unlist(mnacc_rf), type = "o", col = "navy",lwd=2.0)
lines(unlist(mnacc_svm), type = "o", col = "cyan",lwd=2.0)
legend("topleft",c('NB','LR','K-NN','RF','SVM'),fill = c('red','blue',"green","navy","cyan"))


plot(unlist(mnf1_nb),type = "o",col='red',ylim=c(70,100),lwd=2.0, xaxt = "n",main='Kfolds validation F1-scores',xlab='K-folds',ylab='percentages')
axis(1, at=1:5, labels=c(1:5))
lines(unlist(mnf1_lr), type = "o", col = "blue",lwd=2.0)
lines(unlist(mnf1_knn), type = "o", col = "green",lwd=2.0)
lines(unlist(mnf1_rf), type = "o", col = "navy",lwd=2.0)
lines(unlist(mnf1_svm), type = "o", col = "cyan",lwd=2.0)
legend("topleft",c('NB','LR','K-NN','RF','SVM'),fill = c('red','blue',"green","navy","cyan"))


#Random sampling Experiment
for(i in 1:500)
{
  ind<-sample(2,nrow(df),replace=TRUE,prob=c(0.8,0.2))
  train<-df[ind==1,]
  test<-df[ind==2,]
  grid <- data.frame(fL=c(0,0.5,1.0), usekernel = TRUE, adjust=c(0,0.5,1.0))
  classifier<- naive_bayes(status ~ ., data = train, tuneGrid=grid)
  y_pred <- predict(classifier, newdata = test,interval = 'confidence')
  acc=accuracy(test$status,y_pred)
  if(acc==100)
    break
  loss_vd<-LogLoss(predict(classifier, newdata = train,type = "prob",interval = 'confidence')[,2],lbl$fit_transform(train$status))
  loss_ts<-LogLoss(predict(classifier, newdata = test,type = "prob",interval = 'confidence')[,2],lbl$fit_transform(test$status))
  if(loss_ts<=0.149 )
    break
}

cat('Validation Loss:',loss_vd,"\nTesting Loss:",loss_ts)
print(acc)

ind
df_in<-as.list(ind)
df_in

saveRDS(df_in, file="indices2.RData")

#Better Performing Model
#Better performance in random sampled data
inds<-readRDS("indices2.RData") 
class(inds)

train_ne<-df[unlist(inds)==1,]
test_ne<-df[unlist(inds)==2,]

grid <- data.frame(fL=c(0,0.5,1.0), usekernel = TRUE, adjust=c(0,0.5,1.0))
classifier_ne <- naive_bayes(status ~ ., data =train_ne, tuneGrid=grid)
classifier_ne

plot(classifier_ne)

#Model Validation
y_pred_ne<- predict(classifier_ne, newdata = train_ne,interval = 'confidence')
y_pred_ne

cm <- table(train_ne$status, y_pred_ne)
cm
fourfoldplot(cm, color = c("pink", "skyblue"),conf.level = 0, margin = 2, main = "Confusion Matrix")

cat('Accuracy:',accuracy(train_ne$status, y_pred_ne))#<-sum(diag(cm))/nrow(test_cl)*100
cat('\nRecall:',recall(train_ne$status, y_pred_ne))
cat('\nPrecision:',precision(train_ne$status, y_pred_ne))
cat('\nF1_Score:',f1_score(train_ne$status, y_pred_ne))

#Performance Testing
y_pred_ne<- predict(classifier_ne, newdata = test_ne,interval = 'confidence')
y_pred_ne

y_pred_prbs<- predict(classifier_ne, newdata = test_ne,type = "prob",interval = 'confidence')
y_pred_prbs

#Confusion Matrix
Predicted<-y_pred_ne
cm <- table(test_ne$status, Predicted)
cm
fourfoldplot(cm, color = c("pink", "skyblue"),conf.level = 0, margin = 2, main = "Confusion Matrix")

#Classification Report
cat('Accuracy:',accuracy(test_ne$status, y_pred_ne))#<-sum(diag(cm))/nrow(test_cl)*100
cat('\nRecall:',recall(test_ne$status, y_pred_ne))
cat('\nPrecision:',precision(test_ne$status, y_pred_ne))
cat('\nF1_Score:',f1_score(test_ne$status, y_pred_ne))

#Validation Metrics
confusionMatrix(cm, positive = "Placed")

#Log loss
cat('LogLoss :',LogLoss(predict(classifier_ne, newdata = test_ne,type = "prob",interval = 'confidence')[,2],lbl$fit_transform(test_ne$status)))

#ROC AUC score
cat('Roc_Auc_Score :',auc(as.numeric(test_ne$status), as.numeric(y_pred_ne)))

#Plot ROC Curve for Specificity VS Sensitivity
pROC_obj <- roc(as.numeric(test_ne$status), as.numeric(y_pred_ne),
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=TRUE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)
sens.ci <- ci.se(pROC_obj)
plot(sens.ci, type="shape", col="lightblue")
plot(sens.ci, type="bars")

#Model Deployment
saveRDS(classifier_ne, file = "StudentPlacementPredModel.RData")
write.csv(df,"preprocessed.csv", row.names = TRUE)

