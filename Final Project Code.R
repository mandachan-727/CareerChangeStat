library(dplyr)
library(tidyr)
library(ROSE)
library(leaps)
library(MASS)
library(glmnet)
library(pROC)
train <- read.csv('/Users/amandatran0727/Desktop/DATA 300/Final Dataset/aug_train.csv')
#convert null from 'gender" to 'Other'
train$gender[train$gender == ''] <- "Other"
#convert null from 'enrolled_university" to 'no_enrollment'
train$enrolled_university[train$enrolled_university == ''] <- "no_enrollment"
#binary "relevant_experience"
train$relevent_experience <- ifelse(train$relevent_experience == "Has relevent experience", 1, 0)
#group 'experience' into bigger groups: <10, 10-15, >15
train$experience <- ifelse(train$experience == "<1" | as.numeric(train$experience) < 10, "<10", 
                          ifelse(as.numeric(train$experience) >= 10 & as.numeric(train$experience) <= 15, "10-15", 
                                 ifelse(as.numeric(train$experience) > 15, ">15", NA)))
#convert null from 'major_discipline" to 'Other'
train$major_discipline[train$major_discipline == ''] <- "Other"
#group 'company_size' into bigger groups
train$company_size <- ifelse(train$company_size == "<10" | train$company_size == "10/49" |  train$company_size == "50-99", "Small", 
                           ifelse(train$company_size == "100-500" | train$company_size == "500-999" |  train$company_size == "1000-4999", "Mid-size", 
                                  ifelse(train$company_size == "5000-9999" | train$company_size == "10000+", "Large", NA)))
#convert null in 'company_type'
train$company_type[train$company_type == ''] <- "Other"
#re-categorize last_new_job: 0 (no) and 1 (yes)
train$last_new_job <- ifelse(train$last_new_job == "never" | train$last_new_job == "", "0", "1")
train$last_new_job <- as.numeric(train$last_new_job)

#omit remaining NAs:
train <- na.omit(train)

#make binary dummies for 'gender', 'enrolled_university', 'education_level','major_discipline", 'experience','company_size','company_type'
#removing irrelevant columns:
gender_ <- as.factor(train$gender)
gender_ <- model.matrix(~0 + gender_)

Uni_Enrollment_ <- as.factor(train$enrolled_university)
Uni_Enrollment_ <- model.matrix(~0 + Uni_Enrollment_)

Edu_Level_ <- as.factor(train$education_level)
Edu_Level_ <- model.matrix(~0 + Edu_Level_)

Major_ <- as.factor(train$major_discipline)
Major_ <- model.matrix(~0 + Major_)

experience_ <- as.factor(train$experience)
experience_ <- model.matrix(~0 + experience_)

company_size_ <- as.factor(train$company_size)
company_size_ <- model.matrix(~0 + company_size_)

company_type_ <- as.factor(train$company_type)
company_type_ <- model.matrix(~0 + company_type_)

train <- cbind(train [, -c(1,2,4,6,7,8,9,10,11)], gender_[ , -1], Uni_Enrollment_[ ,-1], Edu_Level_[ ,-1], Major_[ ,-1], experience_[ ,-1], company_size_[ ,-1], company_type_[ ,-1])

colnames(train) <- gsub("\\s", "_", colnames(train))
colnames(train) <- gsub("-", "_", colnames(train))

choo <- sample(1:nrow(train), 0.8*nrow(train))
chooset <- train[choo, ]

validation <- setdiff(1:nrow(train),  choo)
validationset <- train[validation, ]

#logistic regression:
log.model <- glm(target~., data = chooset, family = 'binomial')

#predict w/ log.model:
predictTest <- predict(log.model, newdata = validationset, 
                       type = "response")
pred_class <- ifelse(predictTest >= 0.5, 1, 0)
conf_mat <- table(validationset$target, pred_class)
sum(diag(conf_mat)) / sum(conf_mat)

#oversampling:

oversampled <- ovun.sample(target~ city_development_index + relevent_experience + last_new_job + training_hours + target + gender_Male + gender_Other + Uni_Enrollment_no_enrollment + Uni_Enrollment_Part_time_course + Edu_Level_Graduate + Edu_Level_High_School + Edu_Level_Masters + Edu_Level_Phd + Edu_Level_Primary_School + Major_Business_Degree + Major_Humanities + Major_No_Major + Major_Other + Major_STEM + experience_>15 + experience_10_15 + company_size_Mid_size + company_size_Small + company_type_Funded_Startup + company_type_NGO + company_type_Other + company_type_Public_Sector + company_type_Pvt_Ltd, data = train, method = 'over', seed = 123)$data

# ov logistic regression
choo <- sample(1:nrow(oversampled), 0.8*nrow(oversampled))
chooset <- oversampled[choo, ]

validation <- setdiff(1:nrow(oversampled),  choo)
validationset <- oversampled[validation, ]

ov_model <- glm(target~., data = chooset, family = 'binomial')
predictTest <- predict(ov_model, newdata = validationset, 
                       type = "response")
pred_class <- ifelse(predictTest >= 0.5, 1, 0)

ov_confmat <- table(pred_class, validationset$target)
taccuracy_os <- sum(diag(ov_confmat)) / sum(ov_confmat)
taccuracy_os

#undersampling:

undersampled <- ovun.sample(target ~ ., data = train, method = 'under', seed = 123)$data

# ud logistic regression
choo <- sample(1:nrow(undersampled), 0.8*nrow(undersampled))
chooset <- undersampled[choo, ]

validation <- setdiff(1:nrow(undersampled),  choo)
validationset <- undersampled[validation, ]

ud_model <- glm(target~., data = chooset, family = 'binomial')
predictTest <- predict(ud_model, newdata = validationset, 
                       type = "response")
pred_class <- ifelse(predictTest >= 0.5, 1, 0)

ud_confmat <- table(pred_class, validationset$target)
taccuracy_us <- sum(diag(ud_confmat)) / sum(ud_confmat)
taccuracy_us

#best subset selection: 
bss.model <- regsubsets(target~., data = chooset, nvmax = 7)
summary(bss.model)$rsq
data.frame(
  Adj.R2 = which.max(summary(bss.model)$adjr2),
  BIC = which.min(summary(bss.model)$bic),
  Cp = which.min(summary(bss.model)$cp)
)

#bss model: 
log.model <- glm(target~Edu_Level_High_School + city_development_index + relevent_experience + company_size_Mid_size + company_type_Public_Sector, data = chooset, family = 'binomial')

#predict w/ bss.model:
predictTest <- predict(log.model, newdata = validationset, 
                       type = "response")
pred_class <- ifelse(predictTest >= 0.5, 1, 0)
conf_mat <- table(validationset$target, pred_class)
sum(diag(conf_mat)) / sum(conf_mat)

#ridge:
#a
cv_ridge <- cv.glmnet(x, chooset$target, family = 'binomial', alpha = 0)

best_lambda_ridge <- cv_ridge$lambda.min
best_lambda_ridge
cv_ridge.mod <- glmnet(x, chooset$target, family = 'binomial', alpha = 0, lambda = best_lambda_ridge)

#predict with ridge:
predict_ridge <- predict(cv_ridge.mod, newx = data.matrix(validationset), type = 'response', s = best_lambda_ridge)
ridge_pred_class <- ifelse(predict_ridge >= 0.5, 1, 0)

#roc curves:
roc_bss <- roc(validationset$target, predictTest)
roc_ridge <- roc(validationset$target, as.numeric(predict_ridge))


plot(roc_bss, col = "blue", print.auc=TRUE, legacy.axes=TRUE, main="ROC Curves for Credit Models")

# Add the remaining ROC curves
lines(roc_ridge, col = "red")

table(validationset$target, ridge_pred_class)
