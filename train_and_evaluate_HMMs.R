setwd("Repos/rnnalpha/")

library(hmm.discnp)
library(scoring)

data <- strsplit(scan("textual_logs/sepsis.txt", what="", sep="\n"), "")
data <- sapply(data, function(x) c(x, "END"))
vocabulary <- unique(unlist(data))

hmm_max_iter <- 200
#hmm_states_range <- 1:20
n_hmm_states <- length(vocabulary)

set.seed(22)
mean_brier_scores <- c()
for (iter in 1:3) {
  train_ids <- sample(1:length(data), round(2*length(data)/3))
  dt_train <- data[train_ids]
  dt_test <- data[-train_ids]
  
  # TODO: find best number of states
  hmm_model <- try(hmm(dt_train, yval=vocabulary, K = n_hmm_states, itmax = hmm_max_iter),silent=F)
  
  brier_scores <- c()
  for (trace in dt_test) {
    
    # empty trace
    likelihoods <- c()
    for (char in vocabulary) {
      loglik <- logLikHmm(c(char), par=hmm_model)
      likelihoods <- c(likelihoods, exp(loglik))
    } 
    likelihoods_norm <- likelihoods / sum(likelihoods)
    y_true <- rep(0, length(vocabulary))
    y_true[vocabulary==trace[1]] <- 1
    brier_scores <- c(brier_scores, mean(brierscore(y_true~likelihoods_norm)))
    
    # prefixes of length 1 until end
    for (event_nr in 2:(length(trace))) {
      prefix <- trace[1:(event_nr-1)]
      likelihoods <- c()
      for (char in vocabulary) {
        prefix_next <- c(prefix, char)
        loglik <- logLikHmm(prefix_next, par=hmm_model)
        likelihoods <- c(likelihoods, exp(loglik))
      } 
      likelihoods_norm <- likelihoods / sum(likelihoods)
      y_true <- rep(0, length(vocabulary))
      y_true[vocabulary==trace[event_nr]] <- 1
      brier_scores <- c(brier_scores, mean(brierscore(y_true~likelihoods_norm)))
    }
  }
  mean_brier_scores <- c(mean_brier_scores, mean(brier_scores))
}

mean_brier_scores
