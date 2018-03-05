# Chapter 3 - Classifying Sentiment of Film Reviews

devtools::install_github("rstudio/keras")
# Not using below install as it causes a problem.  See:
# https://github.com/rstudio/keras/issues/285
# install.packages("keras")
library(keras)
install_keras()

# Collecting the data, reducing to 1000 to save memory
imdb <- dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb
str(train_data[[1]])
train_labels[[1]]
max(sapply(train_data, max))
# Decoding back to English
word_index <- dataset_imdb_word_index()
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index
# Apply translation function over entire list of words in selected sample
decoded_review <- sapply(test_data[[8]], function(index) {
    # Indexes 1-3 are not really used.  Only translate if greater than 3
    word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
    # If the word is not NULL return it, otherwise replace with ?
    if (!is.null(word)) word else "?"
})

# Encoding the integer sequences into a binary matrix
vectorize_sequences <- function(sequences, dimension = 10000) {
    results <- matrix(0, nrow = length(sequences), ncol = dimension)
    for (i in 1:length(sequences))
        # When i = 3, get row 3 from results (which is all 0).
        # Also get the values from sequence 3 (which are the word numbers e.g. 1, 14, 47, 8, etc.)
        # At each of values in the columns of row 3 of results, turn the 0s into 1s
        results[i, sequences[[i]]] <- 1
    results
}
xtrain <- vectorize_sequences(train_data)
xtest <- vectorize_sequences(test_data)
# Each row is now a sequence of 1s and 0s representing a particular reviews.  
# Each column represents a word and if that word was in that review, the row
# will have a 1 instead of a 0
str(xtrain[1,])
# Convert your labels from integer to numeric
ytrain <- as.numeric(train_labels)
ytest <- as.numeric(test_labels)
# The model definition
model <- keras_model_sequential() %>%
    layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")
# Compile the model
model %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("accuracy")
)
# Setting aside a validation set
valIndices <- 1:10000
xVal <- xtrain[valIndices,]
partialXtrain <- xtrain[-valIndices,]
yVal <- ytrain[valIndices]
partialYtrain <- ytrain[-valIndices]
# Training the model
history <- model %>% fit(
    partialXtrain,
    partialYtrain,
    epochs = 20,
    batch_size = 512,
    validation_data = list(xVal, yVal)
)
str(history)
plot(history)
# Training a new model to reduce overfitting
model <- keras_model_sequential() %>% 
    layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>% 
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("accuracy")
)
model %>% fit(xtrain, ytrain, epochs = 4, batch_size = 512)
results <- model %>%
    evaluate(xtest, ytest)
# Using a trained model to predict
model %>% predict(xtest[1:10,])

