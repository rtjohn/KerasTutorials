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
imdb_dir <- "~/Downloads/aclImdb"
train_dir <- file.path(imdb_dir, "train")







