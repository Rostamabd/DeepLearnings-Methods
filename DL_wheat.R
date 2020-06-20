rm(list=ls())

## we set the seed here to tell the program that always start from the same random number. 
set.seed(12345)

### load DeepGS and BGLR packages
library(DeepGS)
library(BGLR)

#========== Loading genotype and phenotype data #########
### replace your data with wheat data

data(wheat)

X <- wheat.X
### Grain yield phenotype
y <- wheat.Y[,1]

############ Scale X and Y matrices #########################
#y <- scale(y)
X <- scale(X)
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

# cross validation set
trainMat <- X[-tst,]
trainPheno <- y[-tst]
validIdx <- sample(1:nrow(trainMat),floor(nrow(trainMat)*0.1))
validMat <- trainMat[validIdx,]
validPheno <- trainPheno[validIdx]
trainMat <- trainMat[-validIdx,]
trainPheno <- trainPheno[-validIdx]
conv_kernel <- c("1*5") ## convolution kernels (fileter shape)
conv_stride <- c("1*3")
conv_num_filter <- c(8)  ## number of filters
pool_act_type <- c("relu") ## active function for next pool
pool_type <- c("max") ## max pooling shape
pool_kernel <- c("1*2") ## pooling shape
pool_stride <- c("1*2") ## number of pool kernerls
fullayer_num_hidden <- c(16,1)
fullayer_act_type <- c("softrelu")
drop_float <- c(0.2,0.2,0.05)
cnnFrame <- list(conv_kernel =conv_kernel,conv_num_filter = conv_num_filter,
                 conv_stride = conv_stride,pool_act_type = pool_act_type,
                 pool_type = pool_type,pool_kernel =pool_kernel,
                 pool_stride = pool_stride,fullayer_num_hidden= fullayer_num_hidden,
                 fullayer_act_type = fullayer_act_type,drop_float = drop_float)

markerImage = paste0("1*",ncol(trainMat))


trainGSmodel <- train_deepGSModel(trainMat = trainMat,trainPheno = trainPheno,
                validMat = validMat,validPheno = validPheno, markerImage = markerImage, 
                cnnFrame = cnnFrame,device_type = "cpu",gpuNum = 1, eval_metric = "rmse",
                num_round = 100,array_batch_size= 32,learning_rate = 0.01,
                momentum = 0.5,wd = 0.00001, randomseeds = 0,initializer_idx = 0.01,
                verbose =TRUE)

        # make predictions based on the trained model

DeepGS_pred <- predict_GSModel(GSModel = trainGSmodel,testMat = X[tst,],markerImage = markerImage )

DeepGS_pred <- DeepGS_pred*sd(y)+mean(y) 
        # # prepare the prediction matrix

        Yhat <- t(DeepGS_pred)


		mse_A[i] <- mean((y[tst]-Yhat)^2)
		predcor_Ap [i] <- cor(y[tst],Yhat)
		predcor_As[i] <- cor(y[tst],Yhat,method ="spearman")

}

Results <-  cbind(predcor_Ap,predcor_As, mse_A) 

### save predictive ability results

write.table(Results,file="DL_output.txt", quote = FALSE, col.names = c("Pred_Corr_Pearson", "Pred_Corr_Spearman","MSE"))

