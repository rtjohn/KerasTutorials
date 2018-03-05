# Chapter 3 - Classifying Newswires

devtools::install_github("rstudio/keras")
# Not using below install as it causes a problem.  See:
# https://github.com/rstudio/keras/issues/285
# install.packages("keras")
library(keras)
install_keras()
# Colelct and Inspect
reuters <- dataset_reuters(num_words = 10000)
c(c(trainData, trainLabels), c(testData, testLabels)) %<-% reuters
length(trainData) == length(trainLabels)
length(testData) == length(testData)
max(sapply(trainData, max))
summary(trainLabels)
summary(testLabels)
# Decoding back to text
wordIndex <- dataset_reuters_word_index()
revWordIndex <- names(wordIndex)
names(revWordIndex) <- wordIndex
decoded <- function(df, item){
    sapply(df[[item]], function(index) {
        word <- if (index >= 3) revWordIndex[[as.character(index - 3)]]
        if (!is.null(word)) word else "?"
        }
        )
}
decoded(trainData, 1)
# Convert the trainDate list into vectors of 1s and 0s
vectorize_sequences <- function(sequences, dimension = 10000) {
    # Create an empty maxtrix with rows = number of samples
    # Columns are equal to the total vocabulary being used
    results <- matrix(0, nrow = length(sequences), ncol = dimension)
    # For each row in the matrix change a columns 0 to a 1 is that column number
    # is contained in the original sample's numbers
    for (i in 1:length(sequences))
        results[i, sequences[[i]]] <- 1
    results
}
xtrain <- vectorize_sequences(trainData)
xtest <- vectorize_sequences(testData)
# Custom function to one-hot encode the labels
to_one_hot <- function(labels, dimension = 46) {
    results <- matrix(0, nrow = length(labels), ncol = dimension)
    for (i in 1:length(labels))
        # We are doing the +1 here because the lables start at 0 which
        # can't be a column index
        results[i, labels[[i]] + 1] <- 1
    results
}
trainLabelsOh <- to_one_hot(trainLabels)
testLabelsOh <- to_one_hot(testLabels)
# We could have just done this:
#trainLabelsOh <- to_categorical(trainLabels)
#testLabelsOh <- to_categorical(testLabels)

# Model Definition
# NOTE - because we have 46 output possibilities, we should never use fewer
# than 46 units in a layer.  If we do we create an information bottleneck
model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 46, activation = "softmax")
# Compile the model
model %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
)
# Creating a validation set
valIndices <- 1:1000
xval <- xtrain[valIndices,]
partialXtrain <- xtrain[-valIndices,]
yval <- trainLabelsOh[valIndices,]
partialYtrain <- trainLabelsOh[-valIndices,]
# Training the Model
history <- model %>% fit(
    partialXtrain,
    partialYtrain,
    epochs = 20,
    batch_size = 512,
    validation_data = list(xval, yval)
)
plot(history) # Model stops gaining acc around epoch 9

# Retraining a model from scratch for 9 epochs
# NOTE - because we have 46 output possibilities, we should never use fewer
# than 46 units in a layer.  If we do we create an information bottleneck
model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 46, activation = "softmax")
model %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
)
history <- model %>% fit(
    partialXtrain,
    partialYtrain,
    epochs = 9,
    batch_size = 512,
    validation_data = list(xval, yval)
)
results <- model %>% evaluate(xtest, testLabelsOh)
results # ~79% accuracy
# What is random chance accuracy?
testLabelsCopy <- testLabels # Make a copy of test labels
testLabelsCopy <- sample(testLabelsCopy) # Random permutation of test labels
# Check % of original labels that match random permutation
length(which(testLabels == testLabelsCopy)) / length(testLabels)
# ~18.5%

# Generate Predictions for New Data
predictions <- model %>% predict(xtest)
dim(predictions) # should have 46 values for each test row
sum(predictions[1,]) # should sum to 1 because it is a probability
which.max(predictions[1,]) # max is predicted class

