# Taking mtcars datset which is fairly small and implemnting various validation
# mthods

# Goal is to predict mpg using the 11 other features
library(keras)
install_keras()
library(dplyr)
library(ggplot2)
df <- mtcars
dim(df)
# Set Seed so that same sample can be reproduced in future also
set.seed(101) 

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

# Simple Hold Out Method ---------------------------------------------
# Training/Validation Split
indices <- sample(1:nrow(xtrain), size = 0.80 * nrow(xtrain))
valxtrain  <- xtrain[-indices, ]
pxtrain <- xtrain[indices, ]
valytrain  <- ytrain[-indices, ]
pytrain <- ytrain[indices, ]

# Build the model
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
model <- buildModel()
history <- model %>% 
    fit(pxtrain, pytrain, epochs = 2000, batch_size = 2, verbose = 0
    )
ggplot(as.data.frame(history), aes(x = epoch, y = value)) + geom_smooth()
valScore <- model %>% evaluate(valxtrain, valytrain, verbose = 0)
#$loss [1] 3.286091
#$mean_absolute_error [1] 1.589169
model <- buildModel()
history <- model %>% 
    fit(xtrain, ytrain, epochs = 2000, batch_size = 2, verbose = 0
    )
testScore <- model %>% evaluate(xtest, ytest, verbose = 0)
#$loss [1] 12.35774
#$mean_absolute_error [1] 3.306389

# K-fold validation------------------------------------------------------------
k <- 4
# Randomly sample all rows from training data
indices <- sample(1:nrow(xtrain))
# Assign each randomized row to k folds
folds <- cut(indices, breaks=k, labels=FALSE)

numEpochs <- 2000
allScores <- c()
for (i in 1:k) {
    cat("processing fold #", i, "\n")
    # For each k, grab the row indices for that = fold #
    valIndices <- which(folds == i, arr.ind = TRUE)
    # Select them in the training data
    valData <- xtrain[valIndices,]
    # Select them in the target data
    valTargets <- ytrain[valIndices]
    # Deselect them in the training data
    pTrainData <- xtrain[-valIndices,]
    # Deselect them in the testing data
    pTrainTargets <- ytrain[-valIndices]
    # Build the model
    model <- buildModel()
    # Fit and evaluate the model
    model %>% fit(pTrainData, pTrainTargets,
                  epochs = numEpochs, batch_size = 2, verbose = 0)
    results <- model %>% evaluate(valData, valTargets, verbose = 0)
    allScores <- c(allScores, results$loss, results$mean_absolute_error)
}
valLoss <- mean(allScores[c(1,3,5,7)])
# loss 11.15362
valMae <- mean(allScores[c(2,4,6,8)])
# MAE 2.318175

# We used the k-folds to decide on tuning information
# Once we are done we can just run on the whole training set
model <- buildModel()
model %>% fit(xtrain, ytrain, epochs = 2000, batch_size = 2, verbose = 0)
result <- model %>% evaluate(xtest, ytest)
result
#$loss [1] 9.835073
#$mean_absolute_error [1] 2.689343

# Iterated K-fold Validation with Shuffling-------------------------------------

# Merge x and y train for easier shuffling
newdat <- cbind(xtrain,ytrain)
allLoss <- c()
allMae <- c()
S <- 4
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
    folds <- cut(indices, breaks=k, labels=FALSE)
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
mean(allLoss)
mean(allMae)

preds <- model %>% predict(xtest)
comp <- as.data.frame(cbind(ytest, round(preds,1)))
colnames(comp) <- c('true mpg', 'pred mpg')
comp
