# Libraries 
library(MASS)
library(ggplot2)
library(tidyverse)
library(corrplot)
library(car)
library(GGally)
library(ROCR)
library(data.table)

# 1. Load Data
data_path <- '/Users/dap45/OneDrive/Desktop/Data Analytics/DA 6813 - Analytics Applications/Assignments/Collaborative Work/Data-Analytics-Applications/securebank_marketing_dataset_200k.csv'
df = fread(data_path) |> as_tibble()
glimpse(df)

# 2. Preprocessing

df2 <- df |>
  # Take care of character variables
  mutate(Gender = factor(Gender),
         Marital_Status = factor(Marital_Status),
         Loan_Repayment_Status = factor(Loan_Repayment_Status),
         Digital_Banking_Usage = factor(Digital_Banking_Usage),
         Campaign_Response = factor(Campaign_Response, levels = c(0, 1))) |>
  # Create age groups & log income from distribution skewness
  mutate(Age_Group = cut(Age, breaks = c(20, 30, 40, 50, 60, 70, 100),
                         labels = c("21-30", "31-40", "41-50", "51-60", "61-70", "71+"),
                         right = F),
         Log_Income = log1p(Income)) |>
  select(-Customer_ID)

# 3. Visualizations

par(mfrow = c(3, 3))
barplot(table(df2$Campaign_Response))
hist(df2$Age)
barplot(table(df2$Gender))
boxplot(table(df2$Marital_Status))
hist(df2$Income)
hist(df2$Account_Balance)
hist(df2$Credit_Score)
boxplot(table(df2$Loan_Repayment_Status))
hist(df2$Transactions_Per_Month)
par(mfrow = c(1, 1))

# Correlation 
df2_num <- dplyr::select_if(df2, is.numeric)
matrix = cor(df2_num)
corrplot(matrix, method = c("number")) 

# 4. Train/Test Split

# Balanced dataset
df2_yes = dplyr::filter(df2, Campaign_Response == 1)
df2_no = dplyr::filter(df2, Campaign_Response == 0)
df2_random = sample_n(df2_no, dim(df2_yes)[1])
# This is from when we saw the unbalance in the histogram earlier
df2_balance = rbind(df2_yes, df2_no_random)

# Set seed
set.seed(123)

# Unbalanced
train_index <- createDataPartition(df2$Campaign_Response, p = 0.7, list = F) 
train <- df[train_index,]
test <- df[-train_index,]

# Balanced
train_index_balance <- createDataPartition(df2$Campaign_Response, p = 0.7, list = F) 
train_balance <- df[train_index_balance,]
test_balance <- df[-train_index_balance,]

# 5. Logistic Regression

formula_lr <- Campaign_Response ~ Age_Group + Log_Income + Account_Balance + Credit_Score + Transactions_Per_Month + Loan_Repayment_Status + Digital_Banking_Usage + Gender + Marital_Status
lr_model <- glm(formula_lr, data = train, family = binomial)
lt_model_balance <- glm(formula_lr, data = train_balance, family = binomial) 

summary(lr_model)
summary(lr_model_balance)

vif(lr_model)
vif(lr_model_balance)

# Unbalanced predictions
test$prob_lr <- predict(lr_model, newdata = test, type = "response")
test$pred_lr <- ifelse(test$prob_lr > 0.5, "1", "0") |>
  factor(levels = c("0", "1"))

# Balanced predictions
test_balance$prob_lr <- predict(lr_model_balance, newdata = test_balance, type = "response")  
test_balance$pred_lr <- ifelse(test_balance$prob_lr > 0.5, "1", "0") |>
  factor(levels = c("0", "1"))

# AUC & Confusion Matrix
# Unbalanced
roc_lr <- roc(as.numeric(as.character(test$Campaign_Response, test$prob_lr)))
auc_lr <- roc_lr$auc
print(paste("Logistic Regression AUC: ", auc_lr))
print(ConfusionMatrix(test$pred_lr, test$Campaign_Response, positive = "1"))

# Balanced
roc_lr_balance <- roc(as.numeric(as.character(test$Campaign_Response, test$prob_lr)))
auc_lr_balance <- roc_lr_balance$auc
print(paste("Logistic Regression AUC: ", auc_lr_balance))
print(ConfusionMatrix(test_balance$pred_lr, test_balance$Campaign_Response, positive = "1"))

# 6. Decision Tree

formula_ml <- Campaign_Response ~ Age_Group + Log_Income + Account_Balance + Credit_Score + Transactions_Per_Month + Loan_Repayment_Status + Digital_Banking_Usage + Gender + Marital_Status
tree_model <- glm(formula_lr, data = train, family = "class")
tree_model_balance <- glm(formula_lr, data = train_balance, family = "class") 

rpart.plot(tree_model, main = "Decision Tree")
rpart.plot(tree_model_balance, main = "Decision Tree")

# Unbalanced predictions
test$prob_tree <- predict(tree_model, newdata = test)[, "1"]
test$pred_tree <- ifelse(test$prob_tree > 0.5, "1", "0") |>
  factor(levels = c("0", "1"))

# Balanced predictions
test_balance$prob_tree <- predict(tree_model_balance, newdata = test)[, "1"]
test_balance$pred_tree <- ifelse(test_balance$prob_tree > 0.5, "1", "0") |>
  factor(levels = c("0", "1"))

# AUC & Confusion Matrix
# Unbalanced
roc_tree <- roc(as.numeric(as.character(test$Campaign_Response, test$prob_tree)))
auc_tree <- roc_tree$auc
print(paste("Logistic Regression AUC: ", auc_tree))
print(ConfusionMatrix(test$pred_tree, test$Campaign_Response, positive = "1"))

# Balanced
roc_tree_balance <- roc(as.numeric(as.character(test$Campaign_Response, test_balance$prob_tree)))
auc_tree_balance <- roc_tree_balance$auc
print(paste("Logistic Regression AUC: ", auc_tree))
print(ConfusionMatrix(test_balance$pred_tree, test_balance$Campaign_Response, positive = "1"))

# 7. Random Forest

# Unbalanced
rf_model <- randomForest(x = model.matrix(formula_ml, data = train)[, -1],
                         y = train$Campaign_Response,
                         ntree = 150,
                         mtry = floor(sqrt(length(all.vars(formula_ml)))),
                         nodesize = 50,
                         importance = T)
varImpPlot(rf_model, main = "Random Forest")

# Balanced
rf_model_balance <- randomForest(x = model.matrix(formula_ml, data = train_balance)[, -1],
                                 y = train_balance$Campaign_Response,
                                 ntree = 150,
                                 mtry = floor(sqrt(length(all.vars(formula_ml)))),
                                 nodesize = 50,
                                 importance = T)
varImpPlot(rf_model_balance, main = "Random Forest")

# Unbalanced predictions
test_rf <- model.matrix(formula_ml, data = test)[, -1]
test$prob_rf <- predict(rf_model, newdata = test_rf, type = "prob")[, "1"]
test$pred_rf <- ifelse(test$prob_rf > 0.5, "1", "0") |>
  factor(levels = c("0", "1"))

# Balanced predictions
test_rf_balance <- model.matrix(formula_ml, data = test_balance)[, -1]
test$prob_rf_balance <- predict(rf_model_balance, newdata = test_rf_balance, type = "prob")[, "1"]
test$pred_rf_balance <- ifelse(test_balance$prob_rf_balance > 0.5, "1", "0") |>
  factor(levels = c("0", "1"))

# AUC & Confusion Matrix
# Unbalanced
roc_rf <- roc(as.numeric(as.character(test$Campaign_Response, test$prob_rf)))
auc_rf <- roc_rf$auc
print(paste("Logistic Regression AUC: ", auc_rf))
print(ConfusionMatrix(test$pred_rf, test$Campaign_Response, positive = "1"))

# Balanced
roc_rf_balance <- roc(as.numeric(as.character(test$Campaign_Response, test_balance$prob_rf_balance)))
auc_rf_balance <- roc_rf_balance$auc
print(paste("Logistic Regression AUC: ", auc_rf_balance))
print(ConfusionMatrix(test_balance$pred_rf_balance, test_balance$Campaign_Response, positive = "1"))

# 8. ROI Simulation

cost_per_contact <- 5
revenue_per_conversion <- 250

simulate_roi <- function(df_probs, prob_col = "prob"){
  n_total <- nrow(df_probs)
  
  # Mass marketing
  mass_contacts <- n_total
  mass_conversions <- sum(as.numeric(as.character(df_probs$Campaign_Response)))
  mass_cost <- mass_contacts * cost_per_contact
  mass_revenue <- mass_conversions * revenue_per_conversion
  mass_roi <- (mass_revenue - mass_cost) / mass_cost
  
  # Top 20%
  top20_n <- ceiling(0.20 * n_total)
  top20 <- df_probs %>% arrange(desc(.data[[prob_col]])) %>% slice(1:top20_n)
  top20_conversions <- sum(as.numeric(as.character(top20$Campaign_Response)))
  top20_cost <- top20_n * cost_per_contact
  top20_revenue <- top20_conversions * revenue_per_conversion
  top20_roi <- (top20_revenue - top20_cost) / top20_cost
  
  # Top 10%
  top10_n <- ceiling(0.10 * n_total)
  top10 <- df_probs %>% arrange(desc(.data[[prob_col]])) %>% slice(1:top10_n)
  top10_conversions <- sum(as.numeric(as.character(top10$Campaign_Response)))
  top10_cost <- top10_n * cost_per_contact
  top10_revenue <- top10_conversions * revenue_per_conversion
  top10_roi <- (top10_revenue - top10_cost) / top10_cost
  
  tibble(scenario = c("Mass (all)", "Target top 20%", "Target top 10%"),
         contacts = c(mass_contacts, top20_n, top10_n),
         conversions = c(mass_conversions, top20_conversions, top10_conversions),
         cost = c(mass_cost, top20_cost, top10_cost),
         revenue = c(mass_revenue, top20_revenue, top10_revenue),
         ROI = c(as.numeric(mass_roi), as.numeric(top20_roi), as.numeric(top10_roi)))
}

roi_lr <- simulate_roi(test |> rename(prob = prob_lr))
roi_lr_bal <- simulate_roi(test_bal |> rename(prob = prob_lr))

roi_tree <- simulate_roi(test |> rename(prob = prob_tree))
roi_tree_bal <- simulate_roi(test_bal |> rename(prob = prob_tree))

roi_rf <- simulate_roi(test |> rename(prob = prob_rf))
roi_rf_bal <- simulate_roi(test_bal |> rename(prob = prob_rf))


print(list(Logistic = roi_lr, 
           BalancedLogistic = roi_lr_bal,
           Tree = roi_tree, 
           BalancedTree = roi_tree_bal,
           RF = roi_rf, 
           BalancedRF = roi_rf_bal))
