rm(list=ls())

## we set the seed here to tell the program that always start from the same random number. 
set.seed(12345)

### load randomForest program
library(randomForest)
library("BGLR")

#========== Loading genotype and phenotype data #########
### replace your data with wheat data

data(wheat)

X <- wheat.X
### Grain yield phenotype
y <- wheat.Y[,1]

### bind genotype and phenotype in one dataframe
Data <- data.frame(y,X)

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
		
		Xtrn <- Data[-tst,-1]
		y_trn <- Data[-tst,1]
		
		Xtst <- Data[tst,-1]
		y_tst <- Data[tst,1]

		RFfit <- randomForest(x=Xtrn, y=y_trn,nodesize=10,ntree=400)

		### Predictive ability

		Yhat <- predict(RFfit, Xtst)

		mse_A[i] <- mean((y_tst-Yhat)^2)
		predcor_Ap [i] <- cor(y_tst,Yhat)
		predcor_As[i] <- cor(y_tst,Yhat,method ="spearman")

}

Results <-  cbind(predcor_Ap,predcor_As, mse_A) 

### save predictive ability results

write.table(Results,file="RF_output.txt", quote = FALSE, col.names = c("Pred_Corr_Pearson", "Pred_Corr_Spearman","MSE"))

