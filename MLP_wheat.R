rm(list=ls())

## we set the seed here to tell the program that always start from the same random number. 
set.seed(12345)

### load BGLR and mxnet packages

#install.packages("Rcpp")
#install.packages("DiagrammeR")
#install.packages("https://s3.ca-central-1.amazonaws.com/jeremiedb/share/mxnet/CPU/3.6/mxnet.zip", repos = NULL)

install.packages("BGLR")

library(mxnet)
library(BGLR)

#========== Loading genotype and phenotype data #########
### replace your data with wheat data

data(wheat)

X <- wheat.X
### Grain yield phenotype
y <- wheat.Y[,1]

#### Creat kinship-Matrix ========######
###========== # scale X and y data ##====
X <- scale(X)
y <- scale(y)

n=dim(X)[1]
######################################################

### this evaluation is based on 10 replications and 5-fold cross-evalution.
reps <- 10
nFolds <- 5
ntest <- n/nFolds

### Jars to save the results
mse_A <- numeric()
predcor_Ap <- numeric()
predcor_As <- numeric()

### begining of evaluation

for (j in 1:reps){
		cat("Replication:", j,"\n")

		### Splitting the data into differents folds

		tst <-sample(1:n,size=ntest,replace=FALSE)

# cross validation sets
train.x <- data.matrix(X[-tst,])
train.y <- y[-tst]
validIdx <- sample(1:nrow(train.x),floor(nrow(train.x)*0.1))
validMat <- train.x[validIdx,]
validPheno <- train.y[validIdx]
trainMat <- train.x[-validIdx,]
trainPheno <- train.y[-validIdx]

test.x <- data.matrix(X[tst,])
eval.data <- list(data=validMat, label=validPheno)

### MLP fram-work for MXnet program

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=32)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
dropout1 <- mx.symbol.Dropout(data = act1 , p = 0.1)

#fc2 <- mx.symbol.FullyConnected(dropout1, name="fc2", num_hidden=16)
#act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
#dropout2 <- mx.symbol.Dropout(data = act2, p = 0.2)

fc3 <- mx.symbol.FullyConnected(dropout1, name="fc3", num_hidden=16)
act3 <- mx.symbol.Activation(fc3, name="relu3", act_type="softrelu")
dropout3 <- mx.symbol.Dropout(data = act3, p = 0.1)

fc4 <- mx.symbol.FullyConnected(dropout3, name="fc4", num_hidden=1)
lro <- mx.symbol.LinearRegressionOutput(fc4, name="output")

mx.set.seed(0)
model <- mx.model.FeedForward.create(array.layout = "rowmajor",lro,X=train.x, y=train.y,eval.data=eval.data,ctx=mx.cpu(),
        num.round=200, array.batch.size=32, learning.rate=0.01, momentum=0.5,wd = 0.00001,initializer = mx.init.uniform(0.01),epoch.end.callback=mx.callback.log.train.metric(100),
        eval.metric=mx.metric.rmse,verbose =TRUE)

DeepGS_pred = predict(model, test.x,array.layout = "rowmajor")

#######Evaluating the performance of the model################### pf = model %>% evaluate(x = X_ts, y = y_ts, verbose = 0)
Yhat=DeepGS_pred*sd(y)+ mean(y)
Yhat <- t(Yhat)

y_tst <- y[tst]*sd(y)+ mean(y)

		mse_A[j] <- mean((y_tst-Yhat)^2)
		predcor_Ap [j] <- cor(y_tst,Yhat)
		predcor_As[j] <- cor(y_tst,Yhat,method ="spearman")

}

Results <-  cbind(predcor_Ap,predcor_As, mse_A) 

### save predictive ability results

write.table(Results,file="MLP_output.txt", quote = FALSE, col.names = c("Pred_Corr_Pearson", "Pred_Corr_Spearman","MSE"))

