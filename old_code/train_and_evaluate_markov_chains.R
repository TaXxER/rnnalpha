setwd("Repos/rnnalpha/")

library(markovchain)
library(scoring)

data <- strsplit(scan("textual_logs/sepsis.txt", what="", sep="\n"), "")
data <- sapply(data, function(x) c("START", x))
data <- sapply(data, function(x) c(x, "END"))
vocabulary <- unique(unlist(data))
vocabulary <- vocabulary[vocabulary!="START"]

set.seed(22)
mean_brier_scores <- c()
for (iter in 1:3) {
  train_ids <- sample(1:length(data), round(2*length(data)/3))
  #dt_train <- data[train_ids]
  #dt_test <- data[-train_ids]
  dt_train <- data
  dt_test <- data
  
  markov_chain <- markovchainFit(dt_train, possibleStates=c("START", vocabulary))$estimate
  markov_chain <- markov_chain[,names(markov_chain)!="START"]
  
  brier_scores <- c()
  for (trace in dt_test) {
    for (event_nr in 1:(length(trace)-1)) {
      state <- as.character(trace[event_nr])
      # TODO optimize this part
      likelihoods <- c()
      for (char in vocabulary) {
        char <- as.character(char)
        val <- markov_chain[state,char]
        #val <- ifelse(char %in% names(markov_chain), markov_chain[state,char], 0)
        likelihoods <- c(likelihoods, val)
      }
      y_true <- rep(0, length(vocabulary))
      y_true[vocabulary==trace[event_nr+1]] <- 1
      brier_scores <- c(brier_scores, mean(brierscore(y_true~likelihoods)))
    }
  }
  mean_brier_scores <- c(mean_brier_scores, mean(brier_scores))
}

mean_brier_scores
