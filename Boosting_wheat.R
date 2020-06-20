rm(list=ls())

options(expressions = 5e5)
## we set the seed here to tell the program that always start from the same random number. 
set.seed(12345)

### load XGboost program
library(xgboost)
library(BGLR)

#========== Loading genotype and phenotype data #########
### replace your data with wheat data

data(wheat)

X <- wheat.X
### Grain yield phenotype
y <- wheat.Y[,1]

### this evaluation is based on 10 replications and 5-fold cross-evalution.
n <- nrow(X)
reps <- 10
nFolds <- 5
ntest <- n/nFolds

### Jars to save the results
mse_A <- numeric()
predcor_Ap <- numeric()
predcor_As <- numeric()

### begining of evaluation

for (i in 1:reps){
		cat("Replication:", i,"\n")

		### Splitting the data into differents folds

		tst <-sample(1:n,size=ntest,replace=FALSE)
		

X1 <- X[-tst,]
X2 <- X[tst,]
### extract training observations
dd <- y[-tst]

## bind them together
data <- data.frame(dd,X1)
colnames(data)[1] <- "GY"

### Validation data set
sample = sample.int(n = nrow(data), size = floor(.9*nrow(data)), replace = F)

train_t = data[sample, ] #just the samples

valid  = data[-sample, ] #everything but the samples

train_y = train_t[,'GY']
train_x = train_t[, names(train_t) !='GY']

valid_y = valid[,'GY']
valid_x = valid[, names(train_t) !='GY']

gb_train = xgb.DMatrix(data = as.matrix(train_x), label = train_y)
gb_valid = xgb.DMatrix(data = as.matrix(valid_x), label = valid_y)

dtest = xgb.DMatrix(data =  as.matrix(X2))

watchlist = list(train=gb_train, test=gb_valid)

bst_slow = xgb.train(data= gb_train, max.depth = 3,eta = 0.1,nthread = 2,nround = 200,watchlist = watchlist,objective = "reg:linear",
                         early_stopping_rounds = 50,print_every_n = 50)


### Predictive ability

Yhat = predict(bst_slow, dtest)



		mse_A[i] <- mean((y[tst]-Yhat)^2)
		predcor_Ap [i] <- cor(y[tst],Yhat)
		predcor_As[i] <- cor(y[tst],Yhat,method ="spearman")

}

Results <-  cbind(predcor_Ap,predcor_As, mse_A) 

### save predictive ability results

write.table(Results,file="GB_output.txt", quote = FALSE, col.names = c("Pred_Corr_Pearson", "Pred_Corr_Spearman","MSE"))

