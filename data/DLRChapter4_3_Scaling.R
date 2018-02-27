# Taking mtcars datset which is fairly small and implementing various
# methods

# Goal is to predict mpg using the 11 other features
library(keras)
install_keras()
library(dplyr)
library(ggplot2)

df <- mtcars
# Set Seed so that same sample can be reproduced in future also
set.seed(101) 
# Training/Testing Split
# Now Selecting 75% of data as sample from total 'n' rows of the data  
sample <- sample.int(n = nrow(df), size = floor(0.7*nrow(df)), replace = F)
train <- df[sample, ]
test  <- df[-sample, ]
# Scaling the data 
mean <- apply(train, 2, mean)
std <- apply(train, 2, sd)
train <- data.frame(scale(train, center = mean, scale = std))
test <- data.frame(scale(test, center = mean, scale = std))
scaleList <- list(scale = attr(train, "scaled:scale"),
                  center = attr(train, "scaled:center"))
rm(mean, std)

# NOTE - Keras expects a matrix as input not a dataframe
xtrain <- data.matrix(select(train, -mpg))
ytrain <- data.matrix(select(train, mpg))
xtest <- data.matrix(select(test, -mpg))
ytest <- data.matrix(select(test, mpg))

# Build the model--------------------------------------------------------------
buildModel <- function(){
    model <- keras_model_sequential() %>% 
        # 10 features in the data, input shape is 10
        layer_dense(units = 64, activation = 'relu', input_shape = 10) %>%
        layer_dense(units = 64, activation = 'relu') %>%
        layer_dense(units = 1)
    model %>% compile(
        optimizer = "rmsprop",
        loss = "mse",
        metrics = c("mae")
    )
}
# Iterated K-fold Validation with Shuffling-------------------------------------
# Merge x and y train for easier shuffling
newdat <- cbind(xtrain,ytrain)
allLoss <- c()
allMae <- c()
S <- 2
for (i in 1:S) {
    # Shuffle data
    shufIndices <- sample(1:nrow(newdat))
    # Grab those indices from xtrain
    sxtrain <- newdat[shufIndices, -11]
    # Grab those indices from ytrain
    sytrain <- newdat[shufIndices, 11]
    
    # Split into k folds
    k <- 4
    # Randomly sample all rows from training data
    indices <- sample(1:nrow(sxtrain))
    # Assign each randomized row to k folds
    folds <- cut(indices, breaks = k, labels = FALSE)
    # Run k-fold
    numEpochs <- 2000
    for (j in 1:k) {
        cat("processing shuffle #", i, "fold #", j, "\n")
        # For each k, grab the row indices for that = fold #
        valIndices <- which(folds == j, arr.ind = TRUE)
        # Select them in the training data
        valData <- sxtrain[valIndices,]
        # Select them in the target data
        valTargets <- sytrain[valIndices]
        # Deselect them in the training data
        pTrainData <- sxtrain[-valIndices,]
        # Deselect them in the testing data
        pTrainTargets <- sytrain[-valIndices]
        # Build the model
        model <- buildModel()
        # Fit and evaluate the model
        model %>% fit(pTrainData, pTrainTargets,
                      epochs = numEpochs, batch_size = 2, verbose = 0)
        results <- model %>% evaluate(valData, valTargets, verbose = 0)
        allLoss <- c(allLoss, results$loss)
        allMae <- c(allMae, results$mean_absolute_error)
    }
}
scaled <- data.frame("Shuffle_Fold" = 1:length(allLoss), "Loss" = allLoss, 
                    "Mae" = allMae)

# Same thing with un-scaled data-----------------------------------------------
df <- mtcars
dim(df)
# Training/Testing Split
# Now Selecting 75% of data as sample from total 'n' rows of the data  
sample <- sample.int(n = nrow(df), size = floor(0.7*nrow(df)), replace = F)
train <- df[sample, ]
test  <- df[-sample, ]
# NOTE - Keras expects a matrix as input not a dataframe
xtrain <- data.matrix(select(train, -mpg))
ytrain <- data.matrix(select(train, mpg))
xtest <- data.matrix(select(test, -mpg))
ytest <- data.matrix(select(test, mpg))

newdat <- cbind(xtrain,ytrain)
allLoss <- c()
allMae <- c()
S <- 2
for (i in 1:S) {
    # Shuffle data
    shufIndices <- sample(1:nrow(newdat))
    # Grab those indices from xtrain
    sxtrain <- newdat[shufIndices, -11]
    # Grab those indices from ytrain
    sytrain <- newdat[shufIndices, 11]
    
    # Split into k folds
    k <- 4
    # Randomly sample all rows from training data
    indices <- sample(1:nrow(sxtrain))
    # Assign each randomized row to k folds
    folds <- cut(indices, breaks = k, labels = FALSE)
    # Run k-fold
    numEpochs <- 2000
    for (j in 1:k) {
        cat("processing shuffle #", i, "fold #", j, "\n")
        # For each k, grab the row indices for that = fold #
        valIndices <- which(folds == j, arr.ind = TRUE)
        # Select them in the training data
        valData <- sxtrain[valIndices,]
        # Select them in the target data
        valTargets <- sytrain[valIndices]
        # Deselect them in the training data
        pTrainData <- sxtrain[-valIndices,]
        # Deselect them in the testing data
        pTrainTargets <- sytrain[-valIndices]
        # Build the model
        model <- buildModel()
        # Fit and evaluate the model
        model %>% fit(pTrainData, pTrainTargets,
                      epochs = numEpochs, batch_size = 2, verbose = 0)
        results <- model %>% evaluate(valData, valTargets, verbose = 0)
        allLoss <- c(allLoss, results$loss)
        allMae <- c(allMae, results$mean_absolute_error)
    }
}
noScale <- data.frame("Shuffle_Fold" = 1:length(allLoss), "Loss" = allLoss, "Mae" = allMae)
allResults <- rbind(scaled, noScale)
allResults <- cbind(allResults, "Labels" = c(rep("Scaled", 8), rep("Not Scaled", 8)))
ggplot(allResults, aes(x = Shuffle_Fold, y = Loss, color = Labels))+
    geom_smooth()
