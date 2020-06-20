rm(list=ls())

## we set the seed here to tell the program that always start from the same random number. 
set.seed(12345)

### load BGLR program
library(BGLR)

#========== Loading genotype and phenotype data #########
### replace your data with wheat data

data(wheat)

X <- wheat.X
### Grain yield phenotype
y <- wheat.Y[,1]

#### Creat kinship-Matrix ========================######
### check your genotype data MAF, I would suggest use MAF>0.05

X <- scale(X, center=TRUE, scale=TRUE)

n <- nrow(X)
m <- ncol(X)

G <- (tcrossprod(X)/m)+(diag(n)*0.00001)

G <- G+(diag(n)*0.004)
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

for (i in 1:reps){
		cat("Replication:", i,"\n")

		### Splitting the data into differents folds

		tst <-sample(1:n,size=ntest,replace=FALSE)

		yNA <- y

		### modify y to yNA
		yNA[tst] <- NA

		#### RKHS-GBLUP model
		ETA<-list(list(K=G, model='RKHS'))

		fmA<-BGLR(y=yNA,ETA=ETA,thin=5, nIter=6000,burnIn=1000,verbose = TRUE,saveAt="Pred_")


		mse_A[i] <- mean((y[tst]-fmA$yHat[tst])^2)
		predcor_Ap [i] <- cor(y[tst],fmA$yHat[tst])
		predcor_As[i] <- cor(y[tst],fmA$yHat[tst],method ="spearman")

}

Results <-  cbind(predcor_Ap,predcor_As, mse_A) 

### save predictive ability results

write.table(Results,file="GBLUP_output.txt", quote = FALSE, col.names = c("Pred_Corr_Pearson", "Pred_Corr_Spearman","MSE"))

