library(foreign)
library(dplyr)
library(mltools)
library(data.table)
library(mice)
library(finalfit)
library(naniar)
library(GGally)
library(missMDA)
library(SuperLearner)
library(readxl)
library(tidyverse)
library(ggplot2)
library(ggvis)
library(corrplot)
library(caret)
library(randomForest)
library(e1071)
library(caTools)
library(car)
library(stringr)
library(ROSE)
library(sjPlot)
library(sjlabelled)
library(sjmisc)
library(ROCR)
library(dplyr)
library(shapr)
#library(unbalanced)
library(groupdata2)

# Imputation --------------------------------------------------------------

data <- read.spss('data/DB_HER2low_20240312.sav', to.data.frame = TRUE, 
                  add.undeclared.levels = 'no', 
                  use.value.labels = F)

# Remove patient code (useless)
data <- data %>% select(-Pt_code)

# Some variables are unrelated like site of biopsy or timing of biopsy
site_timing_biopsy_var <- grep("site|timing", names(data), ignore.case = T)
data <- data %>% select(-all_of(site_timing_biopsy_var))

# Remove sex because there are too few males
data <- data %>% select(-Sex)

# Remove NA in target
data <- data[!(is.na(data$PrimaryBC_HER2_3cat) | is.na(data$Recurrence_phenotype_HER2_3cat)), ]

data_impute <- imputeMCA(data %>% mutate_all(factor), seed=101)
data <- data_impute$completeObs

# Building the target of interest: patients switching from negative to low HER2
data <- one_hot(as.data.table(data), cols = c("PrimaryBC_HER2_3cat", "Recurrence_phenotype_HER2_3cat"))
data$Switch_HER2_low_gain <- ifelse(data$PrimaryBC_HER2_3cat_0 == 1 & data$Recurrence_phenotype_HER2_3cat_1 == 1, 1, 0) %>% as.factor()


# Remove variables that determine the target
data <- data %>% select(-PrimaryBC_HER2_3cat_0, -PrimaryBC_HER2_3cat_1, -PrimaryBC_HER2_3cat_2, 
                        -Recurrence_phenotype_HER2_3cat_0, -Recurrence_phenotype_HER2_3cat_1, -Recurrence_phenotype_HER2_3cat_2)

# Some variables are clinically not meaningful because they are known after the biopsy
recurrence_pheno_var <- grep("Recurrence_phenotype", names(data), ignore.case = T)
data <- data %>% select(-all_of(recurrence_pheno_var))

# back to numeric
data[] <- lapply(data, function(x) as.numeric(as.character(x)))
data_preBoruta <- data

data_preBoruta %>% write.csv("dataset_imbalanced.csv")
                 

# Ensemble modeling -------------------------------------------------------

# Set model formula
target_variable <- data %>% select(starts_with("Switch_")) %>% colnames()
formula_string <- paste0(target_variable, " ~ .") %>% as.formula()

## Boruta feature selection
library(Boruta)
set.seed(101)
boruta <- Boruta(formula_string, data = data, doTrace = 2, maxRuns = 100)
print(boruta)
png("HER2_low_target/boruta.png", width = 600, height = 350)
plot(boruta)
dev.off()
boruta_filtered_vars <- getSelectedAttributes(boruta, withTentative = T)
data <- data %>% select(all_of(boruta_filtered_vars), all_of(target_variable))


set.seed(101)
partitions <- groupdata2::partition(data = data, p = 0.7, cat_col = target_variable)

# train/test partition
data_train <- partitions[[1]] 
data_test <- partitions[[2]]

# ## Sampling to balance training set
data_train_under <- ovun.sample(formula = formula_string, 
                                data = data_train, 
                                method = "both", 
                                N = 700, 
                                seed = 101)$data


# Prepare data
y <- data_train_under$Switch_HER2_low_gain %>% as.character() %>% as.numeric()
ytest <- data_test$Switch_HER2_low_gain

x <- data_train_under %>% select(-target_variable)
xtest <- data_test %>% select(-target_variable)

## Fit the ensemble model
set.seed(1234)
model <- SuperLearner(y,
                      x,
                      family=binomial(),
                      SL.library=list(
                                      #"SL.ranger",
                                      #"SL.rpartPrune",
                                      #"SL.bayesglm",
                                      "SL.xgboost",
                                      "SL.svm"))

predictions <- predict.SuperLearner(model, newdata=xtest, onlySL = T)
conv.preds <- ifelse(predictions$pred>=0.5,1,0) %>% as.vector
(cm <- confusionMatrix(conv.preds %>% as.factor(), ytest %>% as.factor()))

predictions_df <- data.frame(predictions$pred %>% as.double(), ytest)
colnames(predictions_df) <- c("pred", "Class")
predictions_df %>% probably::cal_plot_logistic(Class, pred)
predictions_df %>% probably::cal_plot_logistic(Class, pred, smooth = FALSE)

pred.model <- glm(Class ~ pred, predictions_df, family = "binomial")
preds.glm <- predict(pred.model, predictions_df, type = "response")
predictions_df_glm <- data.frame(preds.glm %>% as.double(), ytest)
colnames(predictions_df_glm) <- c("pred", "Class")
predictions_df_glm %>% probably::cal_plot_windowed(Class, pred, step_size = 0.02)
conv.preds.glm <- ifelse(predictions_df_glm$pred>=0.2,1,0) %>% as.vector
(cm <- confusionMatrix(conv.preds.glm %>% as.factor(), ytest %>% as.factor()))


## Hyperparameter tuning
SL.svm.tune <- function(...){
  SL.svm(..., class.weights = "inverse")
}

SL.xgboost.tune <- function(...){
  SL.xgboost(..., 
             max.depth = 2,
             eta = 1, 
             nthread = 2, 
             nrounds = 2, 
             objective = "binary:logistic",
             max_delta_step = 10
  )
}

set.seed(150)
model.tune <- SuperLearner(y,
                           x,
                           family=binomial(),
                           SL.library=list(
                                           "SL.xgboost.tune",
                                           "SL.xgboost",
                                           "SL.svm"
                                           ))

predictions <- predict.SuperLearner(model.tune, newdata=xtest, onlySL = T)
conv.preds <- ifelse(predictions$pred>=0.5,1,0) %>% as.vector
(cm <- confusionMatrix(conv.preds %>% as.factor(), ytest %>% as.factor(), positive = '1'))
