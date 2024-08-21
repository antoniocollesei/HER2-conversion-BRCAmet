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

# Rimuovere recurrence_phenotype e tentare visceral

## Load data
data <- read.csv2("data/cleaned_imputed_MCA_data_filtered.csv")

# ## eliminate previous Switch
# data <- data %>% select(-starts_with("Switch_"))
# 
# ## binarize age
# data$Eta_anni_50plus <- ifelse(data$Eta_anni >= 50, 1, 0)
# 
# # creare nuovo target
# data$Switch_HER2_low_gain <- ifelse(data$HER2_primitivo_low == 0 & data$HER2_metastasi_low == 1, 1, 0)
# 
# # remove conditional features
# data <- data %>% select(-HER2_primitivo_low, -HER2_metastasi_low, -Eta_anni, -tempo_alla_biopsia)
# 
# # Keep only patients with HER2_primitivo_positivo == 0
# ## TEST
# 
# # federica filter
# columns_prefiltered_by_federica <- c(
#   "ER_metastasi_ER.0", "ER_metastasi_ER.LOW",
#   "ER_metastasi_ER.", "PgR_metastasi",
#   "HER2_metastasi_negativo",
#   "HER2_metastasi_positivo", "Fenotipo_metastasi_puro.TN",
#   "Fenotipo_metastasi_ER.low", "Fenotipo_metastasi_ER.pos.HER2.",
#   "Fenotipo_metastasi_HER2..any.ER", "Adiuvante_antiHER2"
# )
# data <- data %>% select(-all_of(columns_prefiltered_by_federica))


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

# data %>% write.csv(file = "dataset.csv", row.names = F)

## Variance Inflation Factor
# alias <- alias( lm(formula_string, data = data) )
# alias_data <- alias$Complete %>% row.names()
# data <- data %>% select(-alias_data)
# vif_model <- car::vif( lm(formula_string, data = data) )
# vif_df <- data.frame(variable = names(vif_model), vif = vif_model)
# vif_threshold <- 10
# vif_predictors_to_eliminate <- vif_model[vif_model > vif_threshold] %>% names()
# data <- data %>% select(-all_of(vif_predictors_to_eliminate))

# pdf('HER2_low_target/preliminary_analysis/VIF_analysis.pdf')
# ggplot(vif_df, aes(x = reorder(variable, vif), y = vif)) +
#   geom_bar(stat = "identity", fill = "steelblue") +
#   geom_hline(yintercept = vif_threshold, linetype = "dashed", size = 1.5) +
#   labs(title = "VIF Values", y = "VIF Value", x = "Variables") +
#   theme_minimal() +
#   coord_flip()
# dev.off()

# Drop columns with low variance
#NZV <- nearZeroVar(data, saveMetrics = TRUE)
#nzv_predictors <- NZV[NZV[,"zeroVar"] + NZV[,"nzv"] > 0, ] %>% rownames()
#data <- data %>% select(-all_of(nzv_predictors))

## Modeling
# trControl <- trainControl(method = "cv", number = 10)
# 
# # new formula creation
# formula_string <- paste0(target_variable, " ~ ", paste0(boruta_filtered_vars, collapse = ' + ')) %>% as.formula()

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

# data_train_factor <- data_train %>% mutate_all(factor)
# 
# data.rose <- ROSE::ROSE(formula_string, data = data_train_factor, seed = 1, N = 400)$data
# data.rose <- data.rose %>% lapply(., function(x) as.numeric(as.character(x))) %>% as.data.frame()
# data.rose$Switch_HER2_low_gain <- data.rose$Switch_HER2_low_gain %>% as.factor()

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

## SHAP
library(shapr)

# ind_x_explain <- 1:10
# 
# x_explain <- data_test[, predictor_variables]
# 
# # Specifying the phi_0, i.e. the expected prediction without any features
# p <- mean(data_test$Switch_HER2_low_gain %>% as.numeric()) - 1
# 
# # Computing the actual Shapley values with kernelSHAP accounting for feature dependence using
# # the empirical (conditional) distribution approach with bandwidth parameter sigma = 0.1 (default)
# explanation <- explain(
#   model = glm_model,
#   x_explain = x_explain,
#   x_train = data_train_under[ind_x_explain, predictor_variables],
#   approach = "categorical",
#   prediction_zero = p,
#   #n_combinations = 2^(predictor_variables %>% length())
#   n_combinations = 1000
# )
# 
# # Printing the Shapley values for the test data.
# 
# explanation$shapley_values %>% writexl::write_xlsx("shapley_values_glm.xlsx")

#load("shapley_full_explanation_GLM.RData")

# Plot the resulting explanations for observations 1 and 6
#plot(explanation, plot_phi0 = FALSE, index_x_test = c(1), index_x_explain = 1:5)

#save(explanation, file = "shapley_full_explanation_GLM.RData")


# XGBOOST -----------------------------------------------------------------
library(xgboost)

# trainMNX <- useful::build.x(formula_string, data_under, contrasts = FALSE)
# trainMNY <- useful::build.y(formula_string, data_under)
# trainMNX <- useful::build.x(formula_string, data_train_under, contrasts = FALSE)
# trainMNY <- useful::build.y(formula_string, data_train_under)
# testMNX <- useful::build.x(formula_string, data_test, contrasts = FALSE)
# testMNY <- useful::build.y(formula_string, data_test)
# 
# grid_default <- expand.grid(
#   nrounds = 100,
#   max_depth = 6,
#   eta = 0.3,
#   gamma = 0,
#   colsample_bytree = 1,
#   min_child_weight = 1,
#   subsample = 1
# )
# 
# train_control <- caret::trainControl(
#   method = "repeatedcv", # cross-validation
#   number = 5, # with n folds 
#   verboseIter = FALSE
# )
# 
# xgb_base <- caret::train(
#   x = trainMNX,
#   y = trainMNY,
#   trControl = train_control,
#   tuneGrid = grid_default,
#   method = "xgbTree",
#   verbose = TRUE
# )
# 
# print(xgb_base)
# 
# prediction <- predict(xgb_base, testMNX)
# (confusion_matrix_her2_low_gain <- confusionMatrix(prediction, testMNY, positive = '1'))
# 
# ## tuning xgboost
# tune_grid <- expand.grid(
#   nrounds = 100,
#   eta = c(0.2, 0.3, 0.4),
#   max_depth = c(3, 5, 7),
#   gamma = c(0),
#   colsample_bytree = 1,
#   min_child_weight = 1,
#   subsample = 1
# )
# 
# tune_control <- caret::trainControl(
#   method = "cv", # cross-validation
#   #number = 10, # with n folds 
#   verboseIter = FALSE
# )
# 
# xgb_tune <- caret::train(
#   x = trainMNX,
#   y = trainMNY,
#   trControl = tune_control,
#   tuneGrid = tune_grid,
#   method = "xgbTree",
#   verbose = TRUE,
#   nthreads = 4
# )
# 
# tunePred <- predict(xgb_tune, testMNX)
# tunePred_probs <- predict(xgb_tune, testMNX, type = 'prob')
# (confusion_matrix_her2_low_gain <- confusionMatrix(tunePred, testMNY, positive = '1'))
# 
# #xgb_shap <- xgb.plot.shap(data = testMNX, model = xgb_tune$finalModel, top_n = 20, plot = F)
# 
# library(shapviz)
# 
# xgb_shap <- shapviz::shapviz(xgb_tune$finalModel, X_pred = data.matrix(testMNX), X = testMNX)
# sv_importance(xgb_shap, show_numbers = TRUE)
# sv_waterfall(xgb_shap)

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


save.image('MCA_model_xgboost_definitivo_paper20240415.RData')
## Distance recurrence visceral: recidiva a distanza negli organi vitali
## Primary Pheno Luminal: fenotipo malattia primitiva luminale sfavorisce lo switch
