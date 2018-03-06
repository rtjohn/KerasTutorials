# Loading the IMDB data for use with an embedding layer
# Max number of words you want to consider
max_features <- 10000
# Cutoff samples after 20 words (from max features)
maxlen <- 20
imdb <- dataset_imdb(num_words = max_features)
# Loads data as list of integers
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb
# turns list of integers into tensor of shape (samples, maxlen)
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)

# Using an embedding layer and classifier on the IMDB data
model <- keras_model_sequential() %>%
    # Specifies the maximum input length to the embedding layer so you can 
    # later flatten the embedded inputs. 
    # After the embedding layer, the activations have shape (samples, maxlen, 8).
    layer_embedding(input_dim = 10000, output_dim = 8,
                    input_length = maxlen) %>%
    # Flattens the 3D tensor of embeddings into a 2D tensor of shape 
    # (samples, maxlen * 8)
    layer_flatten() %>%
    # Adds the classifier on top
    layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("acc")
)
summary(model)

history <- model %>% fit(
    x_train, y_train,
    epochs = 10,
    batch_size = 32,
    validation_split = 0.2
)

# Putting it all together: from raw text to word embeddings
# Processing the labels of the raw IMDB data
imdb_dir <- "~/DSwork/aclImdb"
train_dir <- file.path(imdb_dir, "train")
labels <- c()
texts <- c()
for (label_type in c("neg", "pos")) {
    label <- switch(label_type, neg = 0, pos = 1)
    dir_name <- file.path(train_dir, label_type)
    for (fname in list.files(dir_name, pattern = glob2rx("*.txt"), 
                             full.names = TRUE)) {
        texts <- c(texts, readChar(fname, file.info(fname)$size))
        labels <- c(labels, label)
    }
}
# Tokenizing the text of the raw IMBD data
library(keras)
# Cut reviews after 100 words
maxlen <- 100
training_samples <- 600
validation_samples <- 10000
# Consider only top 10,000 words in dataset
max_words <- 10000
# Create the tokenzier
tokenizer <- text_tokenizer(num_words = max_words) %>%
    fit_text_tokenizer(texts)
# Create the sequences of tokens
sequences <- texts_to_sequences(tokenizer, texts)
# Grab the index from the tokenizer now that it has run
word_index <- tokenizer$word_index
cat("Found", length(word_index), "unique tokens.\n")
# > Found 88582 unique tokens.
# Make sure the sequences are all the same lenght
data <- pad_sequences(sequences, maxlen = maxlen)
labels <- as.array(labels)
cat("Shape of data tensor:", dim(data), "\n")
cat('Shape of label tensor:', dim(labels), "\n")
# Splits the data into a training set and a validation set, but first shuffles 
# the data, because you’re starting with data in which samples are ordered 
# (all negative first, then all positive)
indices <- sample(1:nrow(data))
# Grab the 200 for training
training_indices <- indices[1:training_samples]
# Grab the next 10000 for validation
validation_indices <- indices[(training_samples + 1):(training_samples + validation_samples)]
x_train <- data[training_indices,]
y_train <- labels[training_indices]
x_val <- data[validation_indices,]
y_val <- labels[validation_indices]
# Parsing the GloVe word-embeddings file
glove_dir = "~/Downloads/glove"
lines <- readLines(file.path(glove_dir, "glove.6B.100d.txt"))
embeddings_index <- new.env(hash = TRUE, parent = emptyenv())
for (i in 1:length(lines)) {
    line <- lines[[i]]
    values <- strsplit(line, " ")[[1]]
    word <- values[[1]]
    embeddings_index[[word]] <- as.double(values[-1])
}
cat("Found", length(embeddings_index), "word vectors.\n")
# Preparring the GloVe word-embeddings matrix
# The number of dimensions we want in our word embedding dimensions
embedding_dim <- 100
embedding_matrix <- array(0, c(max_words, embedding_dim))
for (word in names(word_index)) {
    index <- word_index[[word]]
    if (index < max_words) {
        embedding_vector <- embeddings_index[[word]]
        if (!is.null(embedding_vector))
            # Words not found in the embedding index will be all zeros.
            embedding_matrix[index+1,] <- embedding_vector
    } }
# Model definition
model <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                    input_length = maxlen) %>%
    layer_flatten() %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")
summary(model)
# Loading pretrained word embeddings into the embedding layer
get_layer(model, index = 1) %>%
    set_weights(list(embedding_matrix)) %>%
    freeze_weights()
# Training and evaluation
model %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("acc")
)
history <- model %>% fit(
    x_train, y_train,
    epochs = 20,
    batch_size = 32,
    validation_data = list(x_val, y_val)
)
save_model_weights_hdf5(model, "pre_trained_glove_model.h5")
# Plotting results
plot(history)

# Training the same model without pretrained word embeddings
model <- keras_model_sequential() %>%
    layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                    input_length = maxlen) %>%
    layer_flatten() %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("acc")
)
history <- model %>% fit(
    x_train, y_train,
    epochs = 20,
    batch_size = 32,
    validation_data = list(x_val, y_val)
)
# Tokenizing the data of the test set
test_dir <- file.path(imdb_dir, "test")
labels <- c()
texts <- c()
for (label_type in c("neg", "pos")) {
    label <- switch(label_type, neg = 0, pos = 1)
    dir_name <- file.path(test_dir, label_type)
    for (fname in list.files(dir_name, pattern = glob2rx("*.txt"),
                             full.names = TRUE)) {
        texts <- c(texts, readChar(fname, file.info(fname)$size))
        labels <- c(labels, label)
    }
}
sequences <- texts_to_sequences(tokenizer, texts)
x_test <- pad_sequences(sequences, maxlen = maxlen)
y_test <- as.array(labels)
# Evaluating model on test set
model %>%
    load_model_weights_hdf5("pre_trained_glove_model.h5") %>%
    evaluate(x_test, y_test)



