# Temperature Forecasting 
# Download data-----------------------------------------------------------------
dir.create("~/Downloads/jena_climate", recursive = TRUE)
download.file(
    "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip",
    "~/Downloads/jena_climate/jena_climate_2009_2016.csv.zip"
    ) 
unzip(
    "~/Downloads/jena_climate/jena_climate_2009_2016.csv.zip",
    exdir = "~/Downloads/jena_climate"
    )
# Ingest and Inspect------------------------------------------------------------
library(tibble)
library(readr)
data_dir <- "~/Downloads/jena_climate"
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")
data <- read_csv(fname)
glimpse(data)
# Plotting temperature timeseries
# NOTE - Data is sample every 10 minutes
library(ggplot2)
ggplot(data, aes(x = 1:nrow(data), y = `T (degC)`)) + geom_line()
# Plotting first 10 days of timeseries
ggplot(data[1:1440,], aes(x = 1:1440, y = `T (degC)`)) + geom_line()
# Preparing the data
lookback <- 1440 # use data from last 10 days
steps <- 6 # take only 1 sample per hour
delay <- 144 # target will be 24 hours in the future
# Convert dataframe into martrix of floating-points
data <- data.matrix(data[,-1])
# Normalizing the data
# Compute mean and sd on training only
train_data <- data[1:200000,] # Grab training data
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)
# Building the Generator Function-----------------------------------------------
# Arguments:
# data - original array of data
# lookback - how many timesteps back should be used for input
# delay - how many timesteps in the future the target should be
# min_index - where to start pulling from 
# max_index - when to stop pulling from
# min and max are used to keep segment for validation and for testing
# shuffle - shuffle samples or draw in chronological order
# batch_size - number of samples per batch
# step - sampling frequency in timesteps
generator <- function(data, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 6) {
    # If there is no max_index, set it to all the data minus the horizon (delay)
    if (is.null(max_index)) {
        max_index <- nrow(data) - delay - 1
    }
    # We only want to use 10 days worth to build so we add 
    # lookback to the starting index
    i <- min_index + lookback
    function() {
        if (shuffle == TRUE) {
            # Yields random sample of n=batch_size indexes from within the range
            # we want to start with, wich is min+lookback - max
            rows <- sample(c((min_index+lookback):max_index), size = batch_size)
        } else {
            # If we've gotten to the end of the data...
            if (i + batch_size >= max_index)
                # ...start over by setting i to it's original state
                i <<- min_index + lookback
            # Otherwise, rows is min_index+lookback through 
            # min_index+lookback+batch_size or min_index+lookback+max_index, 
            # whichever is smaller
            rows <- c(i:min(i+batch_size, max_index))
            # Rows now is the indexes fo the next batch so...
            # i becomes current state + rows
            i <<- i + length(rows)
        }
        # Create an empty array of 0s of shape (128, 240, 14)
        samples <- array(0, dim = c(length(rows), # One row for each row 
                                    # One column for each step (timepoint) 
                                    # we're taking from the lookback
                                    lookback / step,
                                    # One dimension for each feature?
                                    dim(data)[[-1]]))
        # Empty vector of shape (128)
        targets <- array(0, dim = c(length(rows)))
        
        for (j in 1:length(rows)) {
            # Grab a value out of rows (these values are indexes)
            # Go back from that value by amount=lookback and start sequence there
            # Keep going all the way to the value 
            # Produce a sequence of values that is the same length as the number
            # of columns in your empty array (lookback/steps)
            # NOTE -  this will produce decimals but decimals are ignored when
            # indexing?!?!? They are unique indices to the left of the decimal 
            # though
            indices <- seq(rows[[j]] - lookback, rows[[j]],
                           length.out = dim(samples)[[2]])
            # Get all the features from the specified rows in data
            # Place them as a row in the empty array
            samples[j,,] <- data[indices,]
            # Targets are the same indices just plus the delay
            # Dimension is 2 because that's the temp dimension
            targets[[j]] <- data[rows[[j]] + delay,2]
        }
        # Original data was of shape (observations, features)
        # New data is of shape (batch_size, observations, features)
        list(samples, targets)
    }
}
# Preparing the training, validation, and test generators
library(keras)
lookback <- 1440
step <- 6
delay <- 144
batch_size <- 128
# Shuffeld training data, from 1 to 200000
train_gen <- generator(
    data,
    lookback = lookback,
    delay = delay,
    min_index = 1,
    max_index = 200000,
    shuffle = TRUE,
    step = step,
    batch_size = batch_size
)
# Non-shuffled validation data from 200001 to 300000
val_gen = generator(
    data,
    lookback = lookback,
    delay = delay,
    min_index = 200001,
    max_index = 300000,
    step = step,
    batch_size = batch_size
)
# Non-shuffled test data from 3000001 onward
test_gen <- generator(
    data,
    lookback = lookback,
    delay = delay,
    min_index = 300001,
    max_index = NULL,
    step=step,
    batch_size = batch_size
)
# How many steps to draw from val_gen in order to see the entire validation set
val_steps <- (300000 - 200001 - lookback) / batch_size
# How many steps to draw from test_gen in order to see the entire test set
test_steps <- (nrow(data) - 300001 - lookback) / batch_size
# Computing a common-sense baseline MAE
evaluate_naive_method <- function() {
    batch_maes <- c()
    for (step in 1:val_steps) {
        # Grab validation data
        c(samples, targets) %<-% val_gen()
        # Second dimension is temperatiure
        preds <- samples[,dim(samples)[[2]],2]
        # Generator supplied 24 hours ahead for targets based on delay being set
        # to 144
        mae <- mean(abs(preds - targets))
        batch_maes <- c(batch_maes, mae)
    }
    print(mean(batch_maes))
}
error <- evaluate_naive_method()
# converting the MAE back to a Celsius
celsius_mae <- error*std[[2]]
# Basic ML Approach-------------------------------------------------------------
# Training and evaluating densely connected model
model <- keras_model_sequential() %>%
    # Input shape is (observations, dimensions)
    layer_flatten(input_shape = c(lookback / step, dim(data)[-1])) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 1)
model %>% compile(
    optimizer = optimizer_rmsprop(),
    loss = "mae"
)
history <- model %>% fit_generator(
    train_gen,
    steps_per_epoch = 500,
    epochs = 20,
    validation_data = val_gen,
    validation_steps = val_steps
)
plot(history)
# The approach above first flattened the timeseries, which removed the 
# notion of time from the input data.

# Training and evaluating a model with layer_gru
model <- keras_model_sequential() %>%
    layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]])) %>%
    layer_dense(units = 1)
model %>% compile(
    optimizer = optimizer_rmsprop(),
    loss = "mae"
)
history <- model %>% fit_generator(
    train_gen,
    steps_per_epoch = 500,
    epochs = 20,
    validation_data = val_gen,
    validation_steps = val_steps
)
plot(history)

