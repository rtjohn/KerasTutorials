# Word-level one-hot encoding (toy example)
samples <- c("The cat sat on the mat.", "The dog ate my homework.")
# Empty list of tokens
token_index <- list()
# go through each sample
for (sample in samples)
    # Get each word by string splitting
    for (word in strsplit(sample, " ")[[1]])
        # If word not in names of list,
        if (!word %in% names(token_index))
            # add it to the list, counting up
            token_index[[word]] <- length(token_index) + 2
# Max number of words we want to consider
max_length <- 10
# Create array of 0s, with shape (samples, max_length, token index size))
results <- array(0, dim = c(length(samples),
                            max_length,
                            max(as.integer(token_index))))

for (i in 1:length(samples)) {
    # grab the sample
    sample <- samples[[i]]
    # create vector of words of max_length
    words <- head(strsplit(sample, " ")[[1]], n = max_length)
    for (j in 1:length(words)) {
        # find the index number of that word in the token_index
        index <- token_index[[words[[j]]]]
        # at that location in results, create a 1
        results[[i, j, index]] <- 1
    }
}

# Character-level one-hot encoding (toy example)
samples <- c("The cat sat on the mat.", "The dog ate my homework.")
ascii_tokens <- c("", sapply(as.raw(c(32:126)), rawToChar)) 
token_index <- c(1:(length(ascii_tokens))) 
names(token_index) <- ascii_tokens
max_length <- 50
results <- array(0, dim = c(length(samples), max_length, length(token_index)))
for (i in 1:length(samples)) {
    sample <- samples[[i]]
    characters <- strsplit(sample, "")[[1]] 
    for (j in 1:length(characters)) {
        character <- characters[[j]]
        results[i, j, token_index[[character]]] <- 1 }
}

# Using Keras for word-level one-hot encoding
library(keras)
samples <- c("The cat sat on the mat.", "The dog ate my homework.")
tokenizer <- text_tokenizer(num_words = 1000) %>%
    fit_text_tokenizer(samples)
sequences <- texts_to_sequences(tokenizer, samples)
one_hot_results <- texts_to_matrix(tokenizer, samples, mode = "binary")
word_index <- tokenizer$word_index
cat("Found", length(word_index), "unique tokens.\n")

# Word-level one-hot encoding with hashing trick (toy example)
install.packages('hashFunction')
library(hashFunction)
samples <- c("The cat sat on the mat.", "The dog ate my homework.")
# Stores the words as vectors of size 1,000. If you have close to 1,000 words 
# (or more), you’ll see many hash collisions, which will decrease the accuracy
# of this encoding method.
dimensionality <- 1000
max_length <- 10
results <- array(0, dim = c(length(samples), max_length, dimensionality))
for (i in 1:length(samples)) {
    sample <- samples[[i]]
    words <- head(strsplit(sample, " ")[[1]], n = max_length)
    for (j in 1:length(words)) {
        # Use hashFunction::spooky.32() to hash the word into a random integer 
        # index between 0 and 1,000
        index <- abs(spooky.32(words[[i]])) %% dimensionality
        results[[i, j, index]] <- 1
    }
}


