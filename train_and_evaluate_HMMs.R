#!/usr/bin/env Rscript

#setwd("rnnalpha/")

if(!require(hmm.discnp)) install.packages("hmm.discnp",repos = "http://cran.us.r-project.org")
if(!require(scoring)) install.packages("scoring",repos = "http://cran.us.r-project.org")

library(hmm.discnp)
library(scoring)

# load the modified hmm function that is able to deal with empty states
#source('hmm_function_modified.R')

args = commandArgs(trailingOnly=TRUE)

data <- strsplit(scan(sprintf("textual_logs/%s", args[1]), what="", sep="\n"), "")
data <- sapply(data, function(x) c(x, "END"))
vocabulary <- unique(unlist(data))

hmm_max_iter_val <- 10000
hmm_max_iter_test <- 50000

set.seed(22)
mean_brier_scores <- c()
for (iter in 1:3) {
  train_ids <- sample(1:length(data), round(2*length(data)/3))
  dt_train <- data[train_ids]
  dt_test <- data[-train_ids]
  
  # split for model selection
  val_ids <- sample(1:length(dt_train), round(0.2*length(dt_train)))
  dt_val_selection <- dt_train[val_ids]
  dt_train_selection <- dt_train[-val_ids]
  
  ### START MODEL SELECTION ###
  best_states_ratio <- NA
  best_reg <- NA
  best_brier_score <- Inf
  
  for (states_ratio in c(0.1, 0.5, 1.0, 2.0, 5.0, 10.0)) {
    n_hmm_states <- as.integer(states_ratio * length(vocabulary))
    
    for (reg in c("PCLL", "L2", "Linf")){

      #hmm_model <- hmm_modified(dt_train_selection, yval=vocabulary, K=n_hmm_states, crit=reg, itmax=hmm_max_iter_val)
      hmm_model <- hmm(dt_train_selection, yval=vocabulary, K=n_hmm_states, crit=reg, itmax=hmm_max_iter_val)
      
      brier_scores <- c()
      for (trace in dt_val_selection) {
        
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
      current_score <- mean(brier_scores)
      print(sprintf("[validation] iter=%s, n_states_ratio=%s, reg=%s, brier_score=%s", iter, states_ratio, reg, current_score))
      if (current_score < best_brier_score) {
        best_brier_score <- current_score
        best_states_ratio <- states_ratio
        best_reg <- reg
      }
    }
  }
  ### END MODEL SELECTION ###
  
  ### START FINAL MODEL TRAINING AND EVALUATION ###
  n_hmm_states <- as.integer(best_states_ratio * length(vocabulary))
  #hmm_model <- hmm_modified(dt_train, yval=vocabulary, K=n_hmm_states, crit=best_reg, itmax=hmm_max_iter_test)
  hmm_model <- hmm(dt_train, yval=vocabulary, K=n_hmm_states, crit=best_reg, itmax=hmm_max_iter_test)
  
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
  current_score <- mean(brier_scores)
  print(sprintf("[test] iter=%s, n_states_ratio=%s, reg=%s, brier_score=%s", iter, best_states_ratio, best_reg, current_score))
  print("")
  mean_brier_scores <- c(mean_brier_scores, current_score)
  ### END FINAL MODEL TRAINING AND EVALUATION ###
  
}

print(mean_brier_scores)
