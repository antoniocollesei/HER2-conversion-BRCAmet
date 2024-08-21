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
source('tree_func.R')

## Load data
data <- read.csv2("data/cleaned_imputed_MCA_data_filtered.csv")

## Make factor
data$Switch_HER2_low_gain <- data$Switch_HER2_low_gain %>% as.factor()

# Set model formula
target_variable <- data %>% select(starts_with("Switch_")) %>% colnames()
formula_string <- paste0(target_variable, " ~ .") %>% as.formula()

## Boruta feature selection
library(Boruta)
set.seed(101)
boruta <- Boruta(formula_string, data = data, doTrace = 2, maxRuns = 100)
print(boruta)
png("HER2_low_target/boruta.png", width = 600, height = 350)
plot(boruta, las = 2)
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
                                N = 650, 
                                seed = 101)$data


png('high_res/sampling_unbalanced_training_set.png', width = 6000, height = 6000, res = 1200)
ggplot(data_train, aes(x = Switch_HER2_low_gain, fill = Switch_HER2_low_gain)) +
  geom_bar(color = "darkgray") +
  labs(x = "Level", y = "Density") +
  scale_x_discrete(labels = c("No Switch", "Switch")) +
  scale_fill_manual(values = c("0" = "#fbbe22", "1" = "#56106e")) +
  theme(legend.position = "none") +
  ylim(0, 700)
dev.off()

png('high_res/sampling_balanced_training_set.png', width = 6000, height = 6000, res = 1200)
ggplot(data_train_under, aes(x = Switch_HER2_low_gain, fill = Switch_HER2_low_gain)) +
  geom_bar(color = "darkgray") +
  labs(x = "Level", y = "Density") +
  scale_x_discrete(labels = c("No Switch", "Switch")) +
  scale_fill_manual(values = c("0" = "#fbbe22", "1" = "#56106e")) +
  theme(legend.position = "none") +
  ylim(0, 700)
dev.off()

png('low_res/sampling_unbalanced_training_set.png', width = 6000, height = 6000, res = 150)
ggplot(data_train, aes(x = Switch_HER2_low_gain, fill = Switch_HER2_low_gain)) +
  geom_bar(color = "darkgray") +
  labs(x = "Level", y = "Density") +
  scale_x_discrete(labels = c("No Switch", "Switch")) +
  scale_fill_manual(values = c("0" = "#fbbe22", "1" = "#56106e")) +
  theme(legend.position = "none") +
  ylim(0, 700)
dev.off()

png('low_res/sampling_balanced_training_set.png', width = 6000, height = 6000, res = 150)
ggplot(data_train_under, aes(x = Switch_HER2_low_gain, fill = Switch_HER2_low_gain)) +
  geom_bar(color = "darkgray") +
  labs(x = "Level", y = "Density") +
  scale_x_discrete(labels = c("No Switch", "Switch")) +
  scale_fill_manual(values = c("0" = "#fbbe22", "1" = "#56106e")) +
  theme(legend.position = "none") +
  ylim(0, 700)
dev.off()

table(data$Switch_HER2_low_gain)
table(data_train$Switch_HER2_low_gain)
table(data_test$Switch_HER2_low_gain)
table(data_train_under$Switch_HER2_low_gain)

# GLM ---------------------------------------------------------------------

set.seed(101)
glm_model <- glm(formula = formula_string,
                         data = data_train_under,
                         family = binomial)

glm_model %>% summary()

prediction <- ifelse(predict(glm_model, newdata = data_test, type = "response") > 0.5, 1, 0) %>% as.factor()
sink(paste0("HER2_low_target/performance_glm/", target_variable, "_confusion_matrix.txt"))
(confusion_matrix_her2_low_gain <- confusionMatrix(prediction, data_test[[target_variable]], positive = '1'))
sink()


pdf(paste0('HER2_low_target/coefficients_glm/coefficients_', target_variable, '.pdf'), height = 30, width = 15)
sjPlot::plot_model(glm_model, 
                   vline.color = "red", 
                   sort.est = T, 
                   show.values = T,
                   title = paste0('Coefficients odds ratio related to: ', target_variable))
dev.off()


# XGBOOST -----------------------------------------------------------------
library(xgboost)

### tuning xgb
library(mlr)
library(imbalance)

# XGBOOST MLR
set.seed(101, "L'Ecuyer")
traintask <- makeClassifTask (data = data_train_under, target = target_variable, positive = '1')
#traintask <- makeClassifTask (data = data.rose, target = target_variable, positive = '1')
testtask <- makeClassifTask (data = data_test, target = target_variable, positive = '1')

lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list( objective="binary:logistic", eval_metric="error", nrounds=100L, eta=0.3)

#set parameter space
params <- makeParamSet( makeDiscreteParam("booster",values = c("gbtree","gblinear")), makeIntegerParam("max_depth",lower = 3L,upper = 10L), makeNumericParam("min_child_weight",lower = 1L,upper = 10L), makeNumericParam("subsample",lower = 0.5,upper = 1), makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

#set resampling strategy
rdesc <- makeResampleDesc("Bootstrap",stratify = T)

# search strategy
ctrl <- makeTuneControlRandom(maxit = 10L)

#set parallel backend
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

#parameter tuning
mytune <- tuneParams(learner = lrn, 
                     task = traintask, 
                     resampling = rdesc, 
                     measures = gpr, 
                     par.set = params, 
                     control = ctrl, 
                     show.info = T)

#set hyperparameters
lrn_tune <- setHyperPars(lrn, par.vals = mytune$x)

#train model
xgmodel <- train(learner = lrn_tune, task = traintask)

#predict model for training
xgpred_train <- predict(xgmodel, traintask, positive = '1')

cm_train <- confusionMatrix(xgpred_train$data$response, xgpred_train$data$truth, positive = '1')
cm_train

#predict model
xgpred <- predict(xgmodel, testtask, positive = '1')

cm <- confusionMatrix(xgpred$data$response, xgpred$data$truth, positive = '1')
cm

xgboost.model <- getLearnerModel(xgmodel, more.unwrap = TRUE)

# plot model
#library(DiagrammeR)
#xgb.plot.tree(model = xgboost.model, trees = 1:3, show_node_id = FALSE)

clean_feature_names <- c("Primary BC histology",  "Primary BC tumor grade", "Primary BC Phenotype: ER+", "Distant recurrence site: visceral",
                         "Distant recurrence site: soft/skin", "Distant recurrence site: liver", "Distant recurrence site: lung", 
                         "Distant recurrence site: unusual", "Distant recurrencesite: GI", "Distant recurrence site: GU")
xgboost.model$feature_names <- clean_feature_names

# Visualize SHAP values
data_test_X <- data_test %>% select(-Switch_HER2_low_gain)
colnames(data_test_X) <- clean_feature_names
xgb_shap <- shapviz::shapviz(xgboost.model, X_pred = data.matrix(data_test_X), X = data_test_X)

library(shapviz)
png('high_res/shap_importance_xgboost.png', width = 6000, height = 2500, res = 1200)
sv_importance(xgb_shap, show_numbers = TRUE, col = '#fbbe22')
dev.off()

png('high_res/shap_beeswarm_xgboost.png', width = 10000, height = 3000, res = 1200)
sv_importance(xgb_shap, show_numbers = TRUE, kind = 'beeswarm')
dev.off()

png('high_res/shap_force_plot.png', width = 10000, height = 6000, res = 1200)
sv_force(xgb_shap)
dev.off()

png('high_res/shap_waterfall_xgboost.png', width = 6000, height = 6000, res = 1200)
sv_waterfall(xgb_shap)
dev.off()

png('low_res/shap_importance_xgboost.png', width = 6000, height = 2500, res = 300)
sv_importance(xgb_shap, show_numbers = TRUE, col = '#fbbe22')
dev.off()

png('low_res/shap_beeswarm_xgboost.png', width = 10000, height = 3000, res = 300)
sv_importance(xgb_shap, show_numbers = TRUE, kind = 'beeswarm')
dev.off()

png('low_res/shap_force_plot.png', width = 10000, height = 6000, res = 300)
sv_force(xgb_shap)
dev.off()

png('low_res/shap_waterfall_xgboost.png', width = 3000, height = 6000, res = 300)
sv_waterfall(xgb_shap)
dev.off()
