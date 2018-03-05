# Chapter 3 - Boston Housing

devtools::install_github("rstudio/keras")
# Not using below install as it causes a problem.  See:
# https://github.com/rstudio/keras/issues/285
# install.packages("keras")
library(keras)
install_keras()
# Collect and Inspect
dataset <- dataset_boston_housing()
c(c(trainData, trainTargets), c(testData, testTargets)) %<-% dataset
str(trainData)
str(trainTargets)
summary(trainData)
summary(testTargets)
 
# Normalize the data
mean <- apply(trainData, 2, mean)
std <- apply(trainData, 2, sd)
trainData <- scale(trainData, center=mean, scale=std)
# NOTE - we are using the training data means to scale the test data
testData <- scale(testData, center=mean, scale=std)
# Function to make model construction easier
buildModel <- function(){
    model <- keras_model_sequential() %>% 
        layer_dense(units = 64, activation = 'relu',
                    input_shape = dim(trainData)[[2]]) %>% 
        layer_dense(units = 64, activation = 'relu') %>% 
        layer_dense(units=1)
    model %>% compile(
        optimizer = "rmsprop",
        loss = "mse",
        metrics = c("mae")
    )
}
# Because we have few training examples and can't do a simple train/validation/
# test split, we will use K-fold validation
# K-fold validation
k <- 4
# Randomly sample all rows from training data
indices <- sample(1:nrow(trainData))
# Assign each randomized row to k folds
folds <- cut(indices, breaks=k, labels=FALSE)

numEpochs <- 100
allScores <- c()
for (i in 1:k){
    cat("processing fold #", i, "\n")
    # For each k, grab the rwo indices that = fold #
    valIndices <- which(folds == i, arr.ind = TRUE)
    # Select them in the training data
    valData <- trainData[valIndices,]
    # Select them in the target data
    valTargets <- trainTargets[valIndices]
    # Deselect them in the training data
    pTrainData <- trainData[-valIndices,]
    # Deselect them in the testing data
    pTrainTargets <- trainTargets[-valIndices]
    # Buidl the model
    model <- buildModel()
    # Fit and evaluate the model
    model %>% fit(pTrainData, pTrainTargets,
                  epochs = numEpochs, batch_size = 1, verbose = 0)
    results <- model %>% evaluate(valData, valTargets, verbose = 0)
    allScores <- c(allScores, results$mean_absolute_error)
}

# Saving the validation logs at each fold and using more epochs
numEpochs <- 500
allMaeHist <- NULL
for (i in 1:k) {
    cat("processing fold #", i, "\n")
    valIndices <- which(folds == i, arr.ind = TRUE)
    valData <- trainData[valIndices,]
    valTargets <- trainTargets[valIndices]
    pTrainData <- trainData[-valIndices,]
    pTrainTargets <- trainTargets[-valIndices]
    
    model <- buildModel()
    
    history <- model %>% fit(
        pTrainData, pTrainTargets,
        validation_data = list(valData, valTargets),
        epochs = numEpochs, batch_size = 1, verbose = 0
    )
    maeHist <- history$metrics$val_mean_absolute_error
    allMaeHist <- rbind(allMaeHist, maeHist)
}
# Building a history of successive eman K-fold validation scores
avgMaeHist <- data.frame(
    epoch = seq(1:ncol(allMaeHist)),
    validation_mae = apply(allMaeHist, 2, mean)
)
library(ggplot2)
ggplot(avgMaeHist, aes(x = epoch, y = validation_mae)) + geom_smooth()
# We used the k-folds to decide get tuning information
# Once we are done we can just rain on the whole trainingset
model <- buildModel()
model %>% fit(trainData, trainTargets,
              epochs = 70, batch_size = 16, verbose = 0)
result <- model %>% evaluate(testData, testTargets)
result
