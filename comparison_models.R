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

load('comparison_models_data.RData')

prediction_glm <- ifelse(predict(glm_model, newdata = data_test, type = "response") > 0.5, 1, 0) %>% as.factor()
cm_glm <- confusionMatrix(prediction_glm, data_test[[target_variable]], positive = '1')

prediction_xgboost <- xgpred$data$response
cm_xgboost <- confusionMatrix(prediction_xgboost, xgpred$data$truth, positive = '1')

# ensemble
pred_SL <- predict.SuperLearner(model.tune, newdata=xtest, onlySL = T)
prediction_ensemble <- ifelse(pred_SL$pred>=0.5,1,0) %>% as.vector
cm_ensemble <- confusionMatrix(prediction_ensemble %>% as.factor(), ytest %>% as.factor(), positive = '1')

# Plot balanced accuracy, sensitivity, specificity for each model
plot_models <- function(cm_glm, cm_xgboost, cm_ensemble) {
  # Balanced accuracy
  ba_glm <- cm_glm$byClass['Balanced Accuracy']
  ba_xgboost <- cm_xgboost$byClass['Balanced Accuracy']
  ba_ensemble <- cm_ensemble$byClass['Balanced Accuracy']
  
  # Sensitivity
  sens_glm <- cm_glm$byClass['Sensitivity']
  sens_xgboost <- cm_xgboost$byClass['Sensitivity']
  sens_ensemble <- cm_ensemble$byClass['Sensitivity']
  
  # Specificity
  spec_glm <- cm_glm$byClass['Specificity']
  spec_xgboost <- cm_xgboost$byClass['Specificity']
  spec_ensemble <- cm_ensemble$byClass['Specificity']
  
  # Create data frame
  df <- data.frame(
    model = c('GLM', 'XGBoost', 'Ensemble'),
    balanced_accuracy = c(ba_glm, ba_xgboost, ba_ensemble),
    sensitivity = c(sens_glm, sens_xgboost, sens_ensemble),
    specificity = c(spec_glm, spec_xgboost, spec_ensemble)
  )
  
  # Reshape data for plotting
  df_long <- df %>%
    pivot_longer(cols = c(balanced_accuracy, sensitivity, specificity), names_to = 'metric', values_to = 'value')
  
  # Plot
  ggplot(df_long, aes(x = metric, y = value, fill = model)) +
    geom_bar(stat = 'identity', position = 'dodge') +
    scale_fill_manual(values = c('GLM' = '#008080', 'XGBoost' = '#fbbe22', 'Ensemble' = '#56106e')) +
    scale_x_discrete(labels = c('balanced_accuracy' = 'Balanced Accuracy', 'sensitivity' = 'Sensitivity', 'specificity' = 'Specificity')) +
    labs(#title = 'Comparison of models',
         x = 'Metric',
         y = 'Value') +
    #theme_minimal() +
    theme(legend.position = 'right')
}

# Example usage with confusion matrices cm_glm, cm_xgboost, cm_ensemble
png('high_res/comparison_models.png', width = 8000, height = 3000, res = 1200)
plot_models(cm_glm, cm_xgboost, cm_ensemble)
dev.off()

png('low_res/comparison_models.png', width = 8000, height = 3000, res = 600)
plot_models(cm_glm, cm_xgboost, cm_ensemble)
dev.off()


