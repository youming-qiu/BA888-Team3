library(tidyverse)
library(ggplot2)
library(tidytext)
library(tm)

# load datasets
all <- read.csv("data/alldata.csv")
dim(all) # 6964x5

#################### data cleaning process

# re-format the location column and add a new column called 'state'
city_trim <- function(s) gsub('([A-z]+, [A-Z]{2})(\ [0-9]{5})', '\\1', s)
all$location <- sapply(all$location, city_trim)
state_extract <- function(s) gsub('^([A-z| ]+, )([A-Z]{2}).*', '\\2', s)
all$state <- sapply(all$location, state_extract)
dim(all) #6964x6
View(all)

job <- all %>% 
  select(position, description)

# remove http and www elements manually
job$description <- gsub("\\bhttp.*\\b","", job$description)
job$description <- gsub("\\bhttps.*\\b","", job$description)
job$description <- gsub("\\bwww.*\\b","", job$description)
head(job$description)

################
jobs.corpus <- Corpus(VectorSource(job))

# preprocess data
jobs.corpus <- tm_map(jobs.corpus, tolower)
as.character(jobs.corpus[[1]]) # examine the changes to job posting #1

jobs.corpus <- tm_map(jobs.corpus, removeNumbers)
jobs.corpus <- tm_map(jobs.corpus, removePunctuation)

for (j in seq(jobs.corpus)) {
  
  jobs.corpus[[j]] <- gsub("/", " ", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub(" c ", " c_plus_plus ", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("causal inference", "causal_inference", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("communication skills", "communication_skills", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("computer science", "computer_science", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("decision trees", "decision_tree", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("decision tree", "decision_tree", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("dimension reduction", "dimension_reduction", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("dimensionality reduction", "dimension_reduction", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("experimental design", "experimental_design", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("external data", "external_data", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("feature engineering", "feature_engineering", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("feature selection", "feature_selection", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("logistic regression", "logistic_regression", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("ml", "machine_learning", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("machine learning", "machine_learning", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("natural language processing", "nlp", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("neural networks", "neural_network", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("neural network", "neural_network", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("parallel processing", "parallel_processing", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("propensity modeling", "propensity_modeling", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub(" r ", " r_program ", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("random forests", "random_forest", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("random forest", "random_forest", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("supervised learning", "supervised_learning", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("svms", "svm", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("support vector machines", "svm", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("support vector machine", "svm", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("text mining", "text_mining", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("time series", "time_series", jobs.corpus[[j]])
  jobs.corpus[[j]] <- gsub("unsupervised learning", "unsupervised_learning", jobs.corpus[[j]])
  
}

remove(j)

jobs.corpus <- tm_map(jobs.corpus, stripWhitespace)
jobs.corpus <- tm_map(jobs.corpus, removeWords, stopwords("english")) # remove commonly-used words (e.g., "is", "what", etc.)

library(SnowballC)
jobs.corpus <- tm_map(jobs.corpus, stemDocument) # Removing common word endings (e.g., “ing”, “es”, “s”)