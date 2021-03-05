# MGSC 661 Final Project

#-------------------------------------------------------------------------------------------------------------------

# Original Dataset: accepted_2007_to_2018Q4.csv  (1.55 GB after decompressing the .gz file)
# The dataset can be downloaded from : https://www.kaggle.com/wordsforthewise/lending-club 

# As the original dataset was quite large, after pre-processing, we took a sample of 20,000 observations, and stored it as a CSV file 
# This CSV file ("loan_clean_final.csv") has been uploaded on MyCourses along with this code

# Running pre-processing code (from lines 17 to 124) took around 15-20 min to process, since it is a huge dataset
# Please skip to line 125 and import "loan_clean_final.csv" directly, in case you wish to work directly with the cleaned dataset. 

# Please run lines 402 to 708 carefully as these models might take around 15-30 min to process. 
# For clustering, however, lines 503 to 546 must be run before. (myforest_3 model)

#-------------------------------------------------------------------------------------------------------------------
'''
# Pre-Processing of Data

# Load libraries
library(dplyr)

# Read csv file
# NOTE: The file is huge and takes around 5-10 min to load
loan = read.csv("D:\\***\\McGill\\Syllabus\\2_Fall Term\\MGSC 661\\Submissions\\6_Final_Project\\Lending_Club\\accepted_2007_to_2018Q4.csv")

# Exploring Columns
names(loan)
# Total 151 variables

# Analyzing Loan Status
loan%>% group_by(loan_status) %>% summarise(n())

#-------------------------------------------------------------------------------------------------------------------

# 1. Take only observations with Fully Paid or Charged Off
attach(loan)
loan$paid = ifelse(loan_status == "Fully Paid",1,0)
loan$charged_off = ifelse(loan_status == "Charged Off",1,0)

# Subset only if the observation is paid or charged off
loan_subset = subset(loan, paid==1 | charged_off==1)

# Dropping the charged off column
loan_subset = loan_subset[-c(153)] # See column number before dropping
attach(loan_subset)

#-------------------------------------------------------------------------------------------------------------------

# 2. Dropping columns with high NA values (if >25% of values are NA, then drop) 
colSums(is.na(loan_subset))

# => 0.25 * 1345310 = 336327
# Columns with more than 336,327 NA values are dropped

df = loan_subset[,colSums(is.na(loan_subset))<336327]

# Total columns removed = 152 - 108 = 44 columns

#-------------------------------------------------------------------------------------------------------------------

# 3. Dropping columns based on intuition
attach(df)
names(df)

remove_columns = c("id", #1
                   "funded_amnt", #3
                   "funded_amnt_inv", #4
                   "sub_grade", #9
                   "emp_title", #10
                   "loan_status", #16
                   "pymnt_plan", #17
                   "url", #18
                   "desc", #19
                   "title", #21
                   "zip_code", #22
                   "fico_range_high", #28
                   "out_prncp", #36 - All values 0
                   "out_prncp_inv", #37 - All values 0
                   "total_pymnt_inv", #39 - Same as total_pymnt
                   "next_pymnt_d", #47 - Blank column
                   "last_fico_range_high", #49 - Since, we have last_fico_range_low in [50]
                   "policy_code", #52 - All values 1
                   "sec_app_earliest_cr_line", #94
                   "hardship_flag", #95
                   "hardship_type", #96
                   "hardship_status", #98
                   "hardship_start_date", #99
                   "hardship_end_date", #100
                   "payment_plan_start_date", #101
                   "hardship_loan_status", #102
                   "debt_settlement_flag_date", #105
                   "settlement_status", #106
                   "settlement date", #107
                   "verification_status_joint",
                   "hardship_reason",
                   "settlement_date"
                   )


df_2 = df[,!(names(df) %in% remove_columns)]
# Total variables removed = 30 (by 108 - 78)

#-------------------------------------------------------------------------------------------------------------------

# 4. Dropping rows with NA values
df_3 = na.omit(df_2)

#-------------------------------------------------------------------------------------------------------------------

# 5. Sampling 20,000 rows
set.seed(1)
loan_clean = df_3[sample(nrow(df_3), 20000), ]

# loan_clean has 20,000 rows and 78 columns

colSums(is.na(loan_clean)) # Just to confirm

# Saving it in a CSV file
write.csv(loan_clean,'loan_clean_final.csv')

#-------------------------------------------------------------------------------------------------------------------
'''

# Load libraries
library(dplyr)

# Importing clean dataset
# If you ran the above code, please clear all variables from the global environment before running the following code

# Read csv file
loan = read_csv("C:/Users/shiva/OneDrive - McGill University/Desktop/MMA/MGSC 661/Deliverables/Projects/loan_clean_final.csv")
# 20,000 observations of 79 variables

#-------------------------------------------------------------------------------------------------------------------

# Data Pre-processing

# 1. Dropping all the features that will not be available to the investor at the time of making the decision to lend the data 
# Some of the features which had a lot of blank values were also dropped 

# List of columns to be dropped 
drop_cols = c("acc_now_delinq",
              "acc_open_past_24mths",
              "avg_cur_bal",
              "bc_open_to_buy",
              "bc_util",
              "chargeoff_within_12_mths",
              "collection_recovery_fee",
              "collections_12_mths_ex_med",
              "debt_settlement_flag",
              "delinq_2yrs",
              "delinq_amnt",
              "disbursement_method",
              "earliest_cr_line",
              "hardship_reason",
              "inq_last_6mths",
              "int_rate",
              "last_credit_pull_d",
              "last_fico_range_low",
              "last_pymnt_amnt",
              "last_pymnt_d",
              "mo_sin_old_il_acct",
              "mo_sin_old_rev_tl_op",
              "mo_sin_rcnt_rev_tl_op",
              "mo_sin_rcnt_tl",
              "mths_since_recent_bc",
              "mths_since_recent_inq",
              "num_accts_ever_120_pd",
              "num_actv_bc_tl",
              "num_actv_rev_tl",
              " num_bc_sats",
              "num_bc_tl",
              "num_il_tl",
              "num_op_rev_tl",
              "num_rev_accts",
              "num_rev_tl_bal_gt_0",
              "num_sats",
              "num_tl_120dpd_2m",
              "num_tl_30dpd",
              "num_tl_90g_dpd_24m",
              "num_tl_op_past_12m",
              "pct_tl_nvr_dlq",
              "percent_bc_gt_75",
              "recoveries",
              "tax_liens",
              "tot_coll_amt",
              "tot_cur_bal",
              "tot_hi_cred_lim",
              "total_bal_ex_mort",
              "total_bc_limit",
              "total_il_high_credit_limit",
              "total_pymnt",
              "total_rec_int",
              "total_rec_late_fee",
              "total_rec_prncp",
              "total_rev_hi_lim",
              "verification_status_joint", 
              "issue_d",
              "grade")

# Removing the features in the above vector 
df_3 = loan[,!(names(loan) %in% drop_cols)]
# Total variables removed = 54 (by 79 - 25)

# Taking average of fico_range and then putting it as average
# The variable fico_range_high was dropped as it was just four points added to fico_score_low 
# The average would be just adding of two points to fico_score_low
df_3$fico_range = df_3$fico_range_low +2

#-------------------------------------------------------------------------------------------------------------------

# 2. Exploring Variables to determine which ones are categorical and numerical 
#converting all the categorical into factors 
#creating dummy variables for the categorical variables, which have a lot of categories 

# (i) Exploring term variable
df_3$term = as.factor(df_3$term)
table(term)
# term has only two categories: 36 months and 60 months


# (ii) Exploring emp_length variable
df_3$emp_length = as.factor(df_3$emp_length)
table(emp_length)

#Since employment length has over 10 categories, we grouped them together in ranges: <1, 1-3, 4-6, 1-10, >10
# empld less than 1 year is reference 
df_3$empld_1to3 = ifelse(df_3$emp_length == " 1 year"|df_3$emp_length == "2 years" | df_3$emp_length == "3 years",1,0)
df_3$empld_4to6 = ifelse(df_3$emp_length == "4 years" | df_3$emp_length == "5 years" | df_3$emp_length == "6 years",1,0)
df_3$empld_7to10 = ifelse(df_3$emp_length == "7 years" | df_3$emp_length == "8 years" | df_3$emp_length == "9 years"  | df_3$emp_length == "10 years",1,0)
df_3$empld_greaterthan10 = ifelse(df_3$emp_length == "10+ years",1,0)

# (iv) Exploring home_ownership variable
df_3$home_ownership = as.factor(df_3$home_ownership)
table(home_ownership)
#categories: Any, Mortgage, Own, Rent

# (v) Exploring verification_status variable
df_3$verification_status = as.factor(df_3$verification_status)
table(verification_status)
#Categories: Not Verified, Source Verified, Verified 

# (vi) Exploring purpose variable
df_3$purpose = as.factor(df_3$purpose)
table(purpose)

# (vii) Exploring initial_list_status variable
df_3$initial_list_status = as.factor(df_3$initial_list_status)
table(initial_list_status)
#categories: fractional(f) and whole(w)

# (viii) Exploring application_type variable
df_3$application_type = as.factor(df_3$application_type)
table(application_type)
#categories: Individual and Joint App

# (ix) Exploring addr_state variable
df_3$addr_state = as.factor(df_3$addr_state)
attach(df_3)

# (xi) Exploring states 
states = df_3 %>% group_by(addr_state) %>% summarise(n())
print(states)

# Grouping states into regions with midwest as reference
northeast = c("ME","MA", "RI", "CT","NH", "VT", "NY", "PA", "NJ", "DE", "MY")
southeast = c("WV", "VA", "KY", "TN", "NC", "SC", "GA", "AL", "MS", "AR", "LA", "FL")
southwest = c("TX", "OK", "NM", "AZ")
west = c("CO","WY", "MT", "ID","WA", "OR", "UT", "NV", "CA", "AK", "HI")

df_3$NE = ifelse(df_3$addr_state %in% northeast, 1, 0)
df_3$SE = ifelse(df_3$addr_state %in% southeast, 1, 0)
df_3$SW = ifelse(df_3$addr_state %in% southwest, 1, 0)
df_3$west = ifelse(df_3$addr_state %in% west, 1, 0)

# Removing all the duplicate categorical variables for which dummies have been created above 
dupl_cols = c("addr_state",
              "fico_range_low",
              "emp_length")


df_4 = df_3[,!(names(df_3) %in% dupl_cols)]
attach(df_4)
# No. of variables removed = 3 (as 34 - 31)

#-------------------------------------------------------------------------------------------------------------------

# 3. Checking for Multi-Collinearity
df_corr = subset(df_4, select = -c(paid) )
df_corr = as.data.frame(lapply(df_corr, as.numeric))
attach(df_corr)

# 1. Correlation matrix - gives numerical values 
library(ggplot2)
options(max.print=1000000)
quantvars = df_corr[,]
corr_matrix = cor(quantvars)
round(corr_matrix,2)

# 2. (i) Heatmap - NOT to be used - Please skip to alternate method below
# Identify dark regions, take their absolute values and verify from correlation matrix - 
# Remove one of those variable if >0.85
library(corrplot)
library("Hmisc")
library(reshape2)

df_corr.cor = cor(df_corr, method = c("pearson"))
df_corr.rcorr = rcorr(as.matrix(df_corr))
df_corr.rcorr

corrplot(df_corr.cor, method = 'color')
melted_cormat = melt(df_corr.cor)


# (ii) Heatmap (alternate method) - to be used
library(corrplot)
library("Hmisc")
library(reshape2)
library(ggplot2)

df_corr.cor = cor(df_corr, method = c("pearson"))
corrplot(df_corr.cor, method = 'color')
melted_cormat = melt(df_corr.cor)

ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile()+
  scale_fill_gradient2(low = "white", mid = "lightgreen", high = "darkgreen",  
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()

# loan_amnt and installment are highly collinear 

# Removing installment 
df_5 = subset(df_4, select = -c(installment))
# 20,000 observations and 29 variables

#-------------------------------------------------------------------------------------------------------------------

# 4. Checking for Outliers for continuous (numerical variable)

# Ensuring that all categorical variables are converted into factor variables 
attach(df_5)
df_5$term = as.factor(df_5$term)
df_5$empld_1to3 = as.factor(df_5$empld_1to3)
df_5$empld_4to6 = as.factor(df_5$empld_4to6)
df_5$empld_7to10 = as.factor(df_5$empld_7to10)
df_5$empld_greaterthan10 = as.factor(empld_greaterthan10)
df_5$home_ownership = as.factor(df_5$home_ownership)
df_5$verification_status = as.factor(df_5$verification_status)
df_5$purpose = as.factor(df_5$purpose)
df_5$initial_list_status = as.factor(df_5$initial_list_status)
df_5$application_type = as.factor(df_5$application_type)
df_5$NE = as.factor(df_5$NE)
df_5$SE = as.factor(df_5$SE)
df_5$SW = as.factor(df_5$SW)
df_5$west = as.factor(df_5$west)

numerical_df= select_if(df_5, is.numeric)
numerical_columns = names(numerical_df)

# Plotting box plots
attach(numerical_df)
for (i in colnames(numerical_df)){
  if (names(numerical_df[i]) %in% numerical_columns){
    boxplot(numerical_df[[i]],pch=20, col=rgb(0, 0.55, 0.2),main=paste("Graph of", names(numerical_df[i])))
  }
}

# Plotting scatter plots
attach(numerical_df)
for (i in colnames(numerical_df)){
  if (names(numerical_df[i]) %in% numerical_columns){
    plot(numerical_df[[i]],pch=20,col=rgb(0, 0.4, 0.2, 0.2),main=paste("Graph of", names(numerical_df[i])))
  }
}

plot(loan_amnt, annual_inc,pch=19,col=rgb(0, 0.4, 0.2, 0.4),ylab = "Annual Income ($)",xlab = "Loan Amount ($)", main=paste("Annual Income vs Loan Amount"))
plot(annual_inc, dti,pch=19,col=rgb(0, 0.4, 0.2, 0.4),ylab = "Debt to Income Ratio",xlab = "Annual Income ($)", main=paste("Debt to Income Ratio vs Annual Income"))

attach(df_5)
df_5 = df_5[-c(3174,4636,18349),]
attach(df_5)

#-------------------------------------------------------------------------------------------------------------------

# 5. Feature Selection using Boruta and then experimenting with random forest as well 
# Classification to be done using: 1. Random Forest, 2. Boosted Forest, 3. Logistic Regression and compare the models
library(tree)
library(rpart)
library(rpart.plot)
library(randomForest)

attach(df_5)

#(i) Boruta - Takes around half an hour to run, thus the following has been commented
#library(Boruta)
#`%notin%` <- Negate(`%in%`)
#boruta_output <- Boruta(paid ~ ., data=na.omit(df_5), doTrace=2)
#boruta_signif <- names(boruta_output$finalDecision[boruta_output$finalDecision %in% c("Confirmed", "Tentative")])
#boruta_not_signif <- names(boruta_output$finalDecision[boruta_output$finalDecision %notin% c("Confirmed", "Tentative")])
#print(boruta_signif)
#print(boruta_not_signif)
#plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance") 

# Boruta suggests removing the following variables
# "empld_4to6"                
# "empld_7to10"               
# "SE"                        
# "SW"                       
# "west"    

# (ii) Random Forest - for feature selection and classification 
# (a) With all predictors
summary(df_5)
attach(df_5)
set.seed(1)
myforest = randomForest(as.factor(paid)~
                          dti +
                          open_acc +
                          pub_rec +
                          revol_bal +
                          revol_util +
                          total_acc +
                          mort_acc +
                          num_bc_sats +
                          pub_rec_bankruptcies +
                          years_earliest_cr_line +
                          loan_amnt +
                          term+
                          empld_1to3 +
                          empld_4to6 +
                          empld_7to10 +
                          empld_greaterthan10 +
                          home_ownership+
                          annual_inc+
                          verification_status+
                          purpose+
                          initial_list_status+
                          application_type +
                          fico_range+
                          NE +
                          SE +
                          SW +
                          west, ntree=500, data=df_5, importance=TRUE, na.action = na.omit)
myforest
# (a) With all predictors
#OOB estimate of  error rate: 20.2%
Confusion matrix:
#  0     1 class.error
#0 295  3828  0.92845016
#1 210 15661  0.01323168

importance(myforest)
varImpPlot(myforest)

# (b) Testing after removing only SW (since it has negative decmeanaccuracy)
set.seed(1)
myforest_2 = randomForest(as.factor(paid)~
                            dti +
                            open_acc +
                            pub_rec +
                            revol_bal +
                            revol_util +
                            total_acc +
                            mort_acc +
                            num_bc_sats +
                            pub_rec_bankruptcies +
                            years_earliest_cr_line +
                            loan_amnt +
                            term+
                            empld_1to3 +
                            empld_4to6 +
                            empld_7to10 +
                            empld_greaterthan10 +
                            home_ownership+
                            annual_inc+
                            verification_status+
                            purpose+
                            initial_list_status+
                            application_type +
                            fico_range+
                            NE +
                            SE +
                            west, ntree=500, data=df_5, importance=TRUE, na.action = na.omit)
myforest_2
#OOB estimate of  error rate: 20.36%
#Confusion matrix:
#  0     1 class.error
#0 303  3820  0.92650982
#1 251 15620  0.01581501

# (c) Testing after removing insignificant variables suggested by Boruta
library(tree)
library(rpart)
library(rpart.plot)
library(randomForest)
attach(df_5)
set.seed(1)
myforest_3 = randomForest(as.factor(paid)~
                            dti +
                            open_acc +
                            pub_rec +
                            revol_bal +
                            revol_util +
                            total_acc +
                            mort_acc +
                            num_bc_sats +
                            pub_rec_bankruptcies +
                            years_earliest_cr_line +
                            loan_amnt +
                            term+
                            empld_1to3+
                            empld_greaterthan10 +
                            home_ownership+
                            annual_inc+
                            verification_status+
                            purpose+
                            initial_list_status+
                            application_type +
                            fico_range+
                            NE, ntree=500, data=df_5, importance=TRUE, na.action = na.omit)
myforest_3
importance(myforest_3)
varImpPlot(myforest_3, col=rgb(0, 0.4, 0.2), pch=19, main = paste("Variable Importance using Random Forest"))
#OOB estimate of  error rate: 20.26%
#Confusion matrix:
#  0     1 class.error
#0 313  3810  0.92408440
#1 241 15630  0.01518493
importance(myforest_3)

# This means that on an average 20.2% of the out of bag observations were mis-classified. 
# Charged off is 0 and fully paid is 1
# OOB is better for myforest_3, hence the results from Boruta can be used for variable selection 

#-------------------------------------------------------------------------------------------------------------------

# Boosted Forest Algorithm
library(gbm)
attach(df_5)
library(caret)
# Splitting the data into train and test 
# Using caret package so that createDataPartition can generate a stratified random split of the data
set.seed (1)
df_idx = createDataPartition(df_5$paid, p = 0.67, list = FALSE)
train_df = df_5[df_idx, ]
test_df = df_5[-df_idx, ]
boosted=gbm(paid~
              dti +
              open_acc +
              pub_rec +
              revol_bal +
              revol_util +
              total_acc +
              mort_acc +
              num_bc_sats +
              pub_rec_bankruptcies +
              years_earliest_cr_line +
              loan_amnt +
              term+
              empld_1to3 +
              empld_greaterthan10 +
              home_ownership+
              annual_inc+
              verification_status+
              purpose+
              initial_list_status+
              application_type +
              fico_range+
              NE ,
            data=train_df,distribution="bernoulli",n.trees=10000, interaction.depth=3)
summary(boosted)

predicted_gbm=predict(boosted, newdata=test_df, n.trees=10000, type = 'response')
predicted_gbm= ifelse(predicted_gbm>=0.5,1,0)

# predicted_gbm
count = 0 
for (i in (1:6599)){
  if (predicted_gbm[i] != test_df$paid[i]){
    count = count +1 
  }
}
error_gbm = count/6599
error_gbm
predicted_gbm = as.factor(predicted_gbm)
test_df$paid = as.factor(test_df$paid)
confusionMatrix(data=predicted_gbm, test_df$paid)
# Error of 0.01585238 when interaction.depth = 4 (on whole dataset)
# Error of 0.06150923 when interaction.depth = 3(on whole dataset)

# Results from cross validation from GBM
# Error of  0.2315502 when interaction.depth = 3(cross validation)
#Reference
#Prediction    0    1
#0  254  474
#1 1092 4778

#Accuracy : 0.7627          
# 95% CI : (0.7522, 0.7729)

#-------------------------------------------------------------------------------------------------------------------

# Logistic Regression with the same components as myforest_3
attach(df_5)
set.seed(1)
logit = glm(factor(paid)~
              dti +
              open_acc +
              pub_rec +
              revol_bal +
              revol_util +
              total_acc +
              mort_acc +
              num_bc_sats +
              pub_rec_bankruptcies +
              years_earliest_cr_line +
              loan_amnt +
              term+
              empld_1to3 +
              empld_greaterthan10 +
              home_ownership+
              annual_inc+
              verification_status+
              purpose+
              initial_list_status+
              application_type +
              fico_range+
              NE , data = train_df,family="binomial")
predicted_logit = predict(logit, newdata=test_df, type = "response")
predicted_logit
predicted_logit= ifelse(predicted_logit>=0.5,1,0)
#predicted_score
counter = 0 
for (i in (1:6599)){
  if (predicted_logit[i] != test_df$paid[i]){
    counter = counter +1 
  }
}
error_logit = counter/6599
error_logit
test_df$paid = as.factor(test_df$paid)
predicted_logit = as.factor(predicted_logit)
confusionMatrix(data=predicted_logit, test_df$paid)

summary(logit)

# Error rate of 0.1997272

#Reference
#Prediction    0    1
#0   94   66
#1 1252 5186

#Accuracy : 0.8002

# Performed k-fold cross validation to double-check accuracy
attach(df_5)
set.seed(1)
ctrl = trainControl(method = "cv", number = 10, savePredictions = TRUE)
mod_fit = train(factor(paid)~
                  dti +
                  open_acc +
                  pub_rec +
                  revol_bal +
                  revol_util +
                  total_acc +
                  mort_acc +
                  num_bc_sats +
                  pub_rec_bankruptcies +
                  years_earliest_cr_line +
                  loan_amnt +
                  term+
                  empld_1to3 +
                  empld_greaterthan10 +
                  home_ownership+
                  annual_inc+
                  verification_status+
                  purpose+
                  initial_list_status+
                  application_type +
                  fico_range+
                  NE , data = train_df, method="glm", family="binomial",
                trControl = ctrl)
prediction = predict(mod_fit, newdata=test_df)
test_df$paid = as.factor(test_df$paid)
confusionMatrix(data=prediction, test_df$paid)


# Error rate of 0.2076072
#Reference
#Prediction    0    1
#0   94   66
#1 1252 5186

#Accuracy : 0.8002 
# Getting results from logistic regression 
summary(logit)
exp(coef(logit$finalModel))

#-------------------------------------------------------------------------------------------------------------------

# 6. Clustering using predictors from the best model, myforest_3:
# Please make sure to run lines 503 to 546 before running the following
important_variables = importance(myforest_3)

# Boruta suggests removing these variables:
# "empld_4to6"                
# "empld_7to10"               
# "SE"                        
# "SW"                       
# "west" 

df_6 = df_5[,-c(23,24,27,28,29)]
# Total variables: 24

# Top 10 Variables selected using DESC from MeanDecreaseAccuracy from myforest_3 model: 
# Refer important_variables variable
# 1. term       - not numeric
# 2. revol_bal
# 3. annual_inc 
# 4. fico_range   
# 5. open_acc
# 6. loan_amnt
# 7. dti
# 8. total_acc
# 9. revol_util

names(df_6)
df_6 = df_6[,c(3,11,5,21,9,1,8,13,12)]
attach(df_6)
# 9 variables

# Re-scaling
library(dplyr)
rescale_df = df_6 %>% 
  mutate(int_rate_scal = scale(int_rate),
         revol_bal_scal = scale(revol_bal),
         annual_inc_scal = scale(annual_inc),
         fico_range_scal = scale(fico_range),
         open_acc_scal = scale(open_acc),
         loan_amnt_scal = scale(loan_amnt),
         dti_scal = scale(dti),
         total_acc_scal = scale(total_acc),
         revol_util_scal = scale(revol_util))  %>% 
  select(-c(int_rate, revol_bal, annual_inc, fico_range, open_acc, loan_amnt, dti, total_acc, revol_util))

# Testing with 5 clusters
pc_cluster = kmeans(rescale_df, 5)

# Finding the Optimal k using elbow method:

# (i) Constructing a function to compute the total within clusters sum of squares
kmean_withinss = function(k) {
  cluster <- kmeans(rescale_df, k)
  return (cluster$tot.withinss)
}

# (ii) Running the function n times (and max no. of clusters = 20)
max_k = 20 # Maximum no. of clusters = 20

# Running the function from 2 to k 
wss = sapply(2:max_k, kmean_withinss)

# (iii) Creating a data frame to plot the graph
elbow = data.frame(2:max_k, wss)

# (iv) Plotting the graph
ggplot(elbow, aes(x = X2.max_k, y = wss), main = paste("Elbow method") ) +
  geom_point(col = "darkgreen") +
  geom_line(col = "darkgreen",size=0.6) +
  scale_x_continuous(breaks = seq(1, 20, by = 1))+theme_minimal()+
  labs(x= "Number of K", y = "Within-Cluster Sum of Squared distance", title ="Elbow Method")+
  theme(plot.title = element_text(hjust = 0.5))

# From the graph, we can see that the optimal k = 6, 
# Since, at k = 6, the curve starts to have a diminishing return

# Re-running with k = 6
set.seed(1)
pc_cluster_2 = kmeans(rescale_df, 6)

# Silhouette Score for k = 6 clusters
library(purrr)
library(cluster)

km.res = kmeans(df_6, centers = 6, nstart = 25)
ss = silhouette(km.res$cluster, dist(df_6))
avg_sil = mean(ss[, 3])
avg_sil
# 0.4218617

# Insights

# Size of the clusters
pc_cluster_2$size	

# Making box plot for distribution of clusters
# distr is the size of each cluster
distr = c(6108, 3095, 177, 2908, 2935, 4774)
cl = c("1","2","3","4","5", "6")

# Plot the bar chart 
barplot(distr,names.arg=cl,xlab="Cluster Number",ylab="Number of borrowers", col =rgb(0,0.4,0.2), pch =19,main=paste(" Cluster Distrubution"))

# Centers of clusters
center = pc_cluster_2$centers
center

# Using heat map to visualize the centers
# install.packages("RColorBrewer")
library(RColorBrewer)
library(tidyr)

# Creating dataset with the cluster number
cluster = c(1:6)
center_df = data.frame(cluster, center)

# Reshaping the data
center_reshape = gather(center_df, features, values, int_rate_scal: revol_util_scal)
head(center_reshape)

# Creating the palette
library(RColorBrewer)
# par(mar=c(3,4,2,2))
# display.brewer.all()

# Creating the palette
hm.palette = colorRampPalette(rev(brewer.pal(5, 'BuGn')),space='Lab')

# Visualizing using Heatmap
ggplot(data = center_reshape, aes(x = features, y = cluster, fill = values)) +
  scale_y_continuous(breaks = seq(1,6, by = 1)) +
  geom_tile(color = "darkgreen") +
  coord_equal() +
  scale_fill_gradientn(colours = rev(hm.palette(90))) +
  theme_minimal()

#-------------------------------------------------------------------------------------------------------------------