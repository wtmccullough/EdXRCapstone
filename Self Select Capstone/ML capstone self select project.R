if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(Matrix)) install.packages("Matrix", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")

library(readr)
library(tidyverse)
library(caret)
library(data.table)
library(ggplot2)

# load binary data set
ortho_dataset <- read_csv("archive/column_2C_weka.csv")
View(ortho_dataset)

ortho_dataset <- ortho_dataset %>% mutate(class = as.factor(class)) %>% rename(pelvic_tilt = 'pelvic_tilt numeric')
summary(ortho_dataset)

# plots of predictors by classification
plot_dat <- ortho_dataset %>% pivot_longer(!class, names_to = "Predictor", values_to = "Values" )
summary_plots <- plot_dat %>% ggplot(aes(Values, fill = class)) +
  geom_histogram() +
  facet_wrap(~Predictor, scales = "free")

summary_plots

#overall means
ortho_means <- ortho_dataset %>% select(-class) %>% sapply(, FUN = mean)
# mean and sd for predictors by classification
# normal mean and sd first
normal_means <- ortho_dataset %>% filter(class== "Normal") %>% select(-class) %>% sapply(, FUN=mean)
normal_sds <- ortho_dataset %>% filter(class== "Normal") %>% select(-class) %>% sapply(, FUN=sd)

# abnormal
abnormal_means <- ortho_dataset %>% filter(class== "Abnormal") %>% select(-class) %>% sapply(, FUN=mean)
abnormal_sds <- ortho_dataset %>% filter(class== "Abnormal") %>% select(-class) %>% sapply(, FUN=sd)

class_summary <- data.frame(Normal_Mean = normal_means,
                            Normal_Sd = normal_sds,
                            Abnormal_Mean = abnormal_means,
                            Abnormal_Sd = abnormal_sds)


set.seed(33)
test_index <- createDataPartition(y = ortho_dataset$class , times = 1, p = 0.25, list = FALSE)
test <- ortho_dataset[test_index,]
train <- ortho_dataset[-test_index,]

# train models to determine which models to further evaluate

models <- c("glm", "lda", "naive_bayes", "knn", "gamLoess", "rf")
fits <- lapply(models, function(model){ 
  print(model)
  train(class ~ ., method = model, data = train)
}) 
names(fits) <- models
preds <- sapply(fits, function(f){ predict(f, newdata = test)}) %>% as.data.frame() 

acc <- data.frame(glm_acc = mean(test$class==preds$glm),
                  lda_acc = mean(test$class==preds$lda),
                  bayes_acc = mean(test$class==preds$naive_bayes),
                  knn_acc = mean(test$class==preds$knn),
                  gamloess_acc = mean(test$class==preds$gamLoess),
                  rf_acc = mean(test$class==preds$rf))

View(acc)

# test voting model with algos >80% accuracy
set.seed(21)

models <- c("glm", "knn")
vote_fits <- lapply(models, function(model){ 
  print(model)
  train(class ~ ., method = model, data = train)
}) 
names(fits) <- models
vote_preds <- sapply(fits, function(f){ predict(f, newdata = test)}) %>% as.data.frame() 

votes <- rowMeans(vote_preds == "Abnormal")

voting_preds <- ifelse(votes >= 0.5, "Abnormal", "Normal") %>% as.factor()
mean(test$class == voting_preds)

# load 3-way classification dataset
multi_class_data <- read_csv("archive/column_3C_weka.csv")
View(multi_class_data)
multi_class_data <- multi_class_data %>% mutate(class = as.factor(class))

# plots of predictors by classification
plot_dat_multi <- multi_class_data %>% pivot_longer(!class, names_to = "Predictor", values_to = "Values" )
summary_plots_multiclass <- plot_dat_multi %>% ggplot(aes(Values, fill = class)) +
  geom_histogram() +
  facet_wrap(~Predictor, scales = "free")

summary_plots_multiclass

# mean and sd for predictors by classification
normal_means <- multi_class_data %>% filter(class== "Normal") %>% select(-class) %>% sapply(, FUN=mean)
normal_sds <- multi_class_data %>% filter(class== "Normal") %>% select(-class) %>% sapply(, FUN=sd)

# hernia
herni_means <- multi_class_data %>% filter(class== "Hernia") %>% select(-class) %>% sapply(, FUN=mean)
herni_sds <- multi_class_data %>% filter(class== "Hernia") %>% select(-class) %>% sapply(, FUN=sd)

#spondylolisthesis
spondy_means <- multi_class_data %>% filter(class== "Spondylolisthesis") %>% select(-class) %>% sapply(, FUN=mean)
spondy_sds <- multi_class_data %>% filter(class== "Spondylolisthesis") %>% select(-class) %>% sapply(, FUN=sd)

class_summary <- data.frame(Normal_Mean = normal_means,
                            Normal_Sd = normal_sds,
                            Hernia_Mean = herni_means,
                            Hernia_Sd = herni_sds,
                            Spondylolisthesis_Mean = spondy_means,
                            Spondylolisthesis_Sd = spondy_sds)

# create test and train partitions
set.seed(33)
test_index_multi <- createDataPartition(y = multi_class_data$class , times = 1, p = 0.25, list = FALSE)
test_multi <- multi_class_data[test_index_multi,]
train_multi <- multi_class_data[-test_index_multi,]
summary(train_multi)

# test performance of a few models

model_list <- c("lda", "naive_bayes", "knn", "gamLoess", "rf")
set.seed(23)
multi_fits <- lapply(model_list, function(model){ 
  print(model)
  train(class ~ ., method = model, data = train_multi)
}) 
names(multi_fits) <- model_list
preds_multi <- sapply(multi_fits, function(f){ predict(f, newdata = test_multi)}) %>% as.data.frame() 
multi_accuracy <- data.frame(lda_acc = mean(test_multi$class == preds_multi$lda),
                             bayes_acc = mean(test_multi$class == preds_multi$naive_bayes),
                             knn_acc = mean(test_multi$class == preds_multi$knn),
                             gam_acc = mean(test_multi$class == preds_multi$gamLoess),
                             rf_acc = mean(test_multi$class == preds_multi$rf)
                             )
View(multi_accuracy)

# knn model, bayes and random forest voting model on multiple classification

vote_model_list <- c("naive_bayes", "knn", "rf")
set.seed(23)
multi_vote_fits <- lapply(vote_model_list, function(model){ 
  print(model)
  train(class ~ ., method = model, data = train_multi)
}) 
names(multi_vote_fits) <- vote_model_list

preds_votes_multi <- sapply(multi_vote_fits, function(f){ predict(f, newdata = test_multi)}) %>% as.data.frame() 

votes <- preds_votes_multi %>% mutate(norm_vote = ifelse(rowSums(preds_votes_multi=="Normal") >= 2, "Normal", 
                                                         ifelse(rowSums(preds_votes_multi=="Hernia") >= 2, "Hernia", 
                                                                ifelse(rowSums(preds_votes_multi=="Spondylolisthesis") >= 2, "Spondylolisthesis", rf))))
mean(votes$norm_vote == test_multi$class)                                     
                                                                                                                     
                                                                                                                     
                                                                                                                     
