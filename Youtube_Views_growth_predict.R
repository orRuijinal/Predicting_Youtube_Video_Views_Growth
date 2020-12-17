library(caret)
library(randomForest)
library(gbm)
library(glmnet)
library(tidyverse)
library(car)
library(Hmisc)
library(pls)

youtube = read.csv("training.csv")

# Convert publish date to extract time

youtube$PublishedDate
time = str_extract(youtube$PublishedDate, "[:digit:]+:")
hour = str_extract(time, "[:digit:]+")
hour = as.numeric(hour)

youtube = youtube[,-c(1,2)]
dim(youtube)


youtube = add_column(youtube, .before  = "Duration", hour)

###########################################Preprocessing


cor(youtube)
zerov = apply(youtube, 2, function(x){length(unique(x)) == 1})
youtube = youtube [, !zerov] # remove zero sd variables

unique(youtube$punc_num_..23)



#remove cnn_0 only one of them has value
table(youtube$cnn_0)
youtube=subset(youtube, select = -cnn_0)
table(youtube$cnn_36)
youtube=subset(youtube, select = -cnn_36)
table(youtube$cnn_65)
youtube=subset(youtube, select = -cnn_65)
youtube$views_2_hours = log(youtube$views_2_hours)




#------------------------

corr_mat <- cor(youtube)
str(youtube)
high_cor_var  = findCorrelation(corr_mat,
                cutoff = 0.75, names = TRUE)
cat(high_cor_var, sep = ", ")

youtube = subset(youtube, select= -c(hog_13, hog_40, hog_108, hog_144, hog_330, 
                                     hog_336, hog_337, hog_364, hog_368, hog_378, 
                                     hog_386, hog_469, hog_512, hog_514, hog_522, 
                                     hog_576, hog_686, hog_697, hog_702, hog_703, 
                                     hog_704, hog_705, hog_724, hog_725, hog_738,
                                     hog_743, hog_746, hog_774, hog_782, cnn_86,
                                     cnn_88, cnn_89, mean_green, sd_green, mean_blue,
                                     punc_num_..24, hog_89, hog_105, hog_106, hog_125, 
                                     hog_138, hog_139, hog_155, hog_156, hog_177, hog_178,
                                     hog_61, hog_132, hog_133, hog_279, hog_287, hog_165,
                                     hog_204, hog_205, hog_375, hog_376, hog_214, hog_215,
                                     hog_271, hog_306, hog_314, hog_499, hog_351, hog_359, 
                                     hog_557, hog_453, hog_783, hog_788, hog_791, hog_825,
                                     cnn_12, cnn_17, cnn_19, mean_pixel_val, sd_pixel_val,
                                     mean_red, punc_num_..8))



# we want to combine those catagorical variables into one feature
# use the aggregate variables .R



# data patitioning 
set.seed(1667)
trainIndex = createDataPartition(youtube$growth_2_6, p = 0.6,
                                 list = FALSE)
yt_training = youtube[trainIndex,]
yt_testing = youtube[-trainIndex,]

head(yt_training)



#implement lasso regression to perform features selection-------------------------------
x = model.matrix(growth_2_6 ~., data= yt_training)[,-1]
y = yt_training$growth_2_6




#GRID
i.exp <- seq(5, -5, length = 100)
grid <- 10^i.exp

lasso.mod = glmnet(x, y, family = "gaussian", alpha = 1,
                   lambda = grid, standardize = TRUE)

plot(lasso.mod, xvar = 'lambda', label = TRUE)


#cv -lasso
set.seed(213)
lasso.cv.output = cv.glmnet(x, y, family = 'gaussian', alpha = 1,
                            lambda = grid, standardize = TRUE,
                            nfolds = 10)
plot(lasso.cv.output)

lasso.cv.output$lambda.min

a = predict(lasso.mod, s = lasso.cv.output$lambda.min, type = "coefficients")
rownames(a)
nrow(a)
a = names(which(a[-1,] == 0))







# remove variables according to lasso
cat(a, sep = ", ")



# with time 0.8 enet


yt_training = subset(yt_training, select = -c(hog_0, hog_1, hog_4, hog_21, hog_81, hog_94, 
                                              hog_117, hog_183, hog_195, hog_242, hog_259,
                                              hog_303, hog_342, hog_350, hog_363, hog_413, 
                                              hog_442, hog_485, hog_495, hog_523, hog_549, 
                                              hog_584, hog_640, hog_651, hog_655, hog_657, 
                                              hog_658, hog_665, hog_666, hog_668, hog_673, 
                                              hog_698, hog_711, hog_719, hog_747, hog_819, 
                                              hog_827, hog_828, hog_849, hog_852, hog_856, 
                                              hog_857, hog_858, hog_860, cnn_20, sd_red, 
                                              max_blue, edge_avg_value, doc2vec_5, doc2vec_18,
                                              punc_num_..2, punc_num_..4, punc_num_..6, punc_num_..10,
                                              punc_num_..12, punc_num_..13, punc_num_..16, punc_num_..18,
                                              punc_num_..22, punc_num_..26, punc_num_..27, Num_Views_Base_low
                                      ))








# Random Forest (Variable importance)
# use recommmend mtry p/3
dim(yt_training)

set.seed(167)
forestfit.m = randomForest(growth_2_6 ~., data =yt_training, mtry = 34, 
                           importance = T, ntree = 500)

forestfit.pred = predict(forestfit.m , newdata = yt_testing) 
sqrt(mean((forestfit.pred - yt_testing$growth_2_6)^2))  # Check the mse


imp = varImpPlot(forestfit.m)
quantile(imp[,1])
quantile(imp[,2])


# we only used the most importance variables (third quantile)
a = rownames(imp)[which(imp[,1] >= 8.226)]  
b = rownames(imp)[which(imp[,2] >= 184.1)]
# a = a[a %in% b]

#a = rownames(imp)[order(imp[,2], decreasing = TRUE)]

# subset the most importance variables
cat(sapply(strsplit(a, "[, ]+"), function(x) { toString(dQuote(x, FALSE))}), sep = ", ")
cat(a, sep = ", ")

yt_training_copy = yt_training
# yt_training = yt_training_copy
yt_training = subset(yt_training, select = c(hour, Duration, views_2_hours, hog_454, cnn_10, 
                                             cnn_25, cnn_68, sd_blue, punc_num_..28, num_words,
                                             num_chars, num_uppercase_chars, num_digit_chars,
                                             Num_Subscribers_Base_low_mid, Num_Subscribers_Base_mid_high, 
                                             Num_Views_Base_mid_high, avg_growth_low, avg_growth_low_mid, 
                                             avg_growth_mid_high, count_vids_low_mid, growth_2_6))

# use the aggregate variables.R!!!!!!! for the TRAINING DATA SET!!!!!!!!!!!!!!!!!!
# TRAINING



num_subscriber <- rep(0,4346)
num_view <- rep(0,4346)
avg_growth <- rep(0,4346)
count_video <- rep(0,4346)

num_subscriber[which(yt_training$Num_Subscribers_Base_low == 1)] <- 1
num_subscriber[which(yt_training$Num_Subscribers_Base_low_mid == 1)] <- 2
num_subscriber[which(yt_training$Num_Subscribers_Base_mid_high == 1)] <- 3
num_subscriber[which(num_subscriber == 0)] <- 4 
table(num_subscriber)

num_view[which(yt_training$Num_Views_Base_low == 1)] <- 1
num_view[which(yt_training$Num_Views_Base_low_mid == 1)] <- 2
num_view[which(yt_training$Num_Views_Base_mid_high == 1)] <- 3
num_view[which(num_view == 0)] <- 4 
table(num_view)

avg_growth[which(yt_training$avg_growth_low == 1)] <- 1
avg_growth[which(yt_training$avg_growth_low_mid == 1)] <- 2
avg_growth[which(yt_training$avg_growth_mid_high == 1)] <- 3
avg_growth[which(avg_growth == 0)] <- 4
table(avg_growth)

count_video[which(yt_training$count_vids_low == 1)] <- 1
count_video[which(yt_training$count_vids_low_mid == 1)] <- 2
count_video[which(yt_training$count_vids_mid_high == 1)] <- 3
count_video[which(count_video == 0)] <- 4
table(count_video)

num_subscriber <- as.factor(num_subscriber)
num_view <- as.factor(num_view)
avg_growth <- as.factor(avg_growth)
count_video <- as.factor(count_video)


yt_training = subset(yt_training, select = -c(Num_Subscribers_Base_low, Num_Subscribers_Base_low_mid, Num_Subscribers_Base_mid_high,
                                              Num_Views_Base_low, Num_Views_Base_low_mid, Num_Views_Base_mid_high ,
                                              avg_growth_low, avg_growth_low_mid , avg_growth_mid_high, 
                                              count_vids_low, count_vids_low_mid, count_vids_mid_high))

dim(youtube)
# add the new columns
yt_training = add_column(yt_training , .before = "growth_2_6", num_subscriber)
yt_training  = add_column(yt_training , .before = "growth_2_6", num_view)
yt_training  = add_column(yt_training , .before = "growth_2_6", avg_growth)
yt_training  = add_column(yt_training , .before = "growth_2_6", count_video)





# now use the aggregate variables.R to convert those low,high,mid into one variables
# Use the aggregate variables.R file to select testing variables!!!!!!!!!!!!!!!!!!!!!!!
# for the testing set -------------------------------------------------------------------------------------------
dim(real_test)
testnum_subscriber <- rep(0,2896)
testnum_view <- rep(0,2896)
testavg_growth <- rep(0,2896)
testcount_video <- rep(0,2896)
testnum_subscriber[which(yt_testing$Num_Subscribers_Base_low == 1)] <- 1
testnum_subscriber[which(yt_testing$Num_Subscribers_Base_low_mid == 1)] <- 2
testnum_subscriber[which(yt_testing$Num_Subscribers_Base_mid_high == 1)] <- 3
testnum_subscriber[which(testnum_subscriber == 0)] <- 4 


testnum_view[which(yt_testing$Num_Views_Base_low == 1)] <- 1
testnum_view[which(yt_testing$Num_Views_Base_low_mid == 1)] <- 2
testnum_view[which(yt_testing$Num_Views_Base_mid_high == 1)] <- 3
testnum_view[which(testnum_view == 0)] <- 4 


testavg_growth[which(yt_testing$avg_growth_low == 1)] <- 1
testavg_growth[which(yt_testing$avg_growth_low_mid == 1)] <- 2
testavg_growth[which(yt_testing$avg_growth_mid_high == 1)] <- 3
testavg_growth[which(testavg_growth == 0)] <- 4


testcount_video[which(yt_testing$count_vids_low == 1)] <- 1
testcount_video[which(yt_testing$count_vids_low_mid == 1)] <- 2
testcount_video[which(yt_testing$count_vids_mid_high == 1)] <- 3
testcount_video[which(testcount_video == 0)] <- 4


testnum_subscriber <- as.factor(testnum_subscriber)
testnum_view <- as.factor(testnum_view)
testavg_growth <- as.factor(testavg_growth)
testcount_video <- as.factor(testcount_video)




yt_testing = subset(yt_testing, select = -c(Num_Subscribers_Base_low, Num_Subscribers_Base_low_mid, Num_Subscribers_Base_mid_high,
                                          Num_Views_Base_low, Num_Views_Base_low_mid, Num_Views_Base_mid_high ,
                                          avg_growth_low, avg_growth_low_mid , avg_growth_mid_high, 
                                          count_vids_low, count_vids_low_mid, count_vids_mid_high))


# add the new columns
yt_testing = add_column(yt_testing , .after = "num_digit_chars", num_subscriber = testnum_subscriber)
yt_testing  = add_column(yt_testing , .after = "num_digit_chars", num_view =testnum_view)
yt_testing  = add_column(yt_testing , .after = "num_digit_chars", avg_growth =testavg_growth)
yt_testing  = add_column(yt_testing , .after = "num_digit_chars", count_video =testcount_video)





set.seed(1697)
23/3
shrink.forestfit = randomForest(growth_2_6 ~., data = yt_training, mtry = 6, ntree = 750,
                           importance = F)
shrink.forestfit.pred = predict(shrink.forestfit, newdata = yt_testing)
sqrt(mean((shrink.forestfit.pred - yt_testing$growth_2_6)^2))
#Indeed. Our rmse decreases! and we see that this works! now we further use CV to select our best mtry!


#try different mtry to see if there's improvement
set.seed(167)
oob_train_control = trainControl(method = "oob",
                                 savePredictions = TRUE)
tunegrid <- expand.grid(mtry= 1:20)
forestfit = train(growth_2_6 ~., data = yt_training, method = "rf",
                  importance = FALSE, trControl = oob_train_control, tuneGrid = tunegrid)
print(forestfit)
forest.pred = predict(forestfit, yt_testing)
sqrt(mean((forest.pred - yt_testing$growth_2_6)^2))


print(forestfit) # use mtry = 19
tunegrid <- expand.grid(mtry= 12)
set.seed(1678)
random.fit = train(growth_2_6 ~., data = training, method = "rf",
                   importance = FALSE, trControl = oob_train_control, tuneGrid = tunegrid)
random.pred22 = predict(random.fit, yt_testing)
sqrt(mean((random.pred22 - yt_testing$growth_2_6)^2))
random.fit










#---------------bagging-----------####
set.seed(155)
bag.fit = randomForest(growth_2_6 ~., data = yt_training, mtry = 20, 
                       importance = F)


bag.pred = predict(bag.fit, yt_testing, ntree = 500)
sqrt(mean((bag.pred - yt_testing$growth_2_6)^2))






###############################################################################################

#------------------------------------BOOSTING---------------------------------------------------
set.seed(1222)
boost_yt = gbm(growth_2_6~. , data = yt_training,
               distribution = "gaussian", interaction.depth = 4,n.trees = 500
               )
summary(boost_yt)

i.exp <- seq(-5, -0.5, by = 0.1)
lambda <- 10^i.exp
test.mse <- rep(NA, length(lambda))
for (i in 1:length(lambda)){
  boost_hit = gbm(growth_2_6 ~., data = yt_training, n.trees = 500,
                  distribution = "gaussian", shrinkage = lambda[i], interaction.depth = 4)
  boost_pred <- predict(boost_hit, newdata = yt_testing,
                        n.trees = 500)
  test.mse[i] <- mean((boost_pred- yt_testing$growth_2_6)^2)
}
plot(lambda, test.mse, type = 'b', xlab= "shrinkage",
     ylab = 'Testing set MSE')

lambda[which.min(test.mse)]

best_boost_yt = gbm(growth_2_6~. , data = yt_training,
               distribution = "gaussian", interaction.depth = 4,n.trees = 500,
               shrinkage = lambda[which.min(test.mse)])


summary(best_boost_yt)


boost.pred = predict(best_boost_yt, newdata = yt_testing, n.trees = 500)
sqrt(mean((boost.pred - yt_testing$growth_2_6)^2))

#-----------------------------Ridge
#RIDGE REGRESS
set.seed(367)

x = model.matrix(growth_2_6 ~. , yt_training)[,-1]
y = yt_training$growth_2_6

i.exp = seq(5, -5, length = 100)
grid = 10^i.exp
ridge.mod = glmnet(x, y , family = "gaussian", alpha = 0, lambda = grid, standardize = TRUE)
plot(lasso.mod, xvar = 'lambda', label = TRUE)

set.seed(368)
cv.ridge = cv.glmnet(x,y, family = "gaussian", alpha = 0 , lambda = grid, standardize = TRUE,
                     nfolds = 10)
cv.ridge$lambda.min


best.ridge.mod = glmnet(x, y , family = "gaussian", alpha = 0, lambda = cv.ridge$lambda.min
                        , standardize = TRUE)


fit  =cv.ridge$glmnet.fit
new_test = model.matrix(growth_2_6~., data= yt_testing)[,-1]
ridge.pred  = predict(fit, s =  cv.ridge$lambda.min, newx = new_test)
sqrt(mean((ridge.pred - testing$growth_2_6)^2)) # 2.83


#-----------------------------LASSO
#LASSO REGRESS
set.seed(1)

x = model.matrix(growth_2_6 ~. , training)[,-1]
y = training$growth_2_6

i.exp = seq(5, -5, length = 100)
grid = 10^i.exp
lasso.mod = glmnet(x, y , family = "gaussian", alpha = 1, lambda = grid, standardize = TRUE)
plot(lasso.mod, xvar = 'lambda', label = TRUE)

cv.lasso = cv.glmnet(x,y, family = "gaussian", alpha = 1 , lambda = grid, standardize = TRUE,
                     nfolds = 10)
cv.lasso$lambda.min


best.lasso.mod = glmnet(x, y , family = "gaussian", alpha = 1, lambda = cv.ridge$lambda.min
                        , standardize = TRUE)


fit  =cv.lasso$glmnet.fit
new_test = model.matrix(growth_2_6~., data= testing)[,-1]
lasso.pred  = predict(fit, s =  cv.lasso$lambda.min,newx = new_test)
sqrt(mean((lasso.pred - testing$growth_2_6)^2))

###################################################################################################
############################################################################################################


#predict the real data
real_test = read.csv("test.csv")
# add time variables
test_time = str_extract(real_test$PublishedDate, "[:digit:]+:")
hour = str_extract(test_time, "[:digit:]+")
hour = (as.numeric(hour))
real_test = real_test[,-c(1,2)]
real_test = add_column(real_test, .before  = "Duration", hour)
real_test = real_test[,names[-21]]

# for the testing set -------------------------------------------------------------------------------------------
dim(real_test)
testnum_subscriber <- rep(0,3105)
testnum_view <- rep(0,3105)
testavg_growth <- rep(0,3105)
testcount_video <- rep(0,3105)
testnum_subscriber[which(real_test$Num_Subscribers_Base_low == 1)] <- 1
testnum_subscriber[which(real_test$Num_Subscribers_Base_low_mid == 1)] <- 2
testnum_subscriber[which(real_test$Num_Subscribers_Base_mid_high == 1)] <- 3
testnum_subscriber[which(testnum_subscriber == 0)] <- 4 


testnum_view[which(real_test$Num_Views_Base_low == 1)] <- 1
testnum_view[which(real_test$Num_Views_Base_low_mid == 1)] <- 2
testnum_view[which(real_test$Num_Views_Base_mid_high == 1)] <- 3
testnum_view[which(testnum_view == 0)] <- 4 


testavg_growth[which(real_test$avg_growth_low == 1)] <- 1
testavg_growth[which(real_test$avg_growth_low_mid == 1)] <- 2
testavg_growth[which(real_test$avg_growth_mid_high == 1)] <- 3
testavg_growth[which(testavg_growth == 0)] <- 4


testcount_video[which(real_test$count_vids_low == 1)] <- 1
testcount_video[which(real_test$count_vids_low_mid == 1)] <- 2
testcount_video[which(real_test$count_vids_mid_high == 1)] <- 3
testcount_video[which(testcount_video == 0)] <- 4


testnum_subscriber <- as.factor(testnum_subscriber)
testnum_view <- as.factor(testnum_view)
testavg_growth <- as.factor(testavg_growth)
testcount_video <- as.factor(testcount_video)




real_test = subset(real_test, select = -c(Num_Subscribers_Base_low, Num_Subscribers_Base_low_mid, Num_Subscribers_Base_mid_high,
                                            Num_Views_Base_low, Num_Views_Base_low_mid, Num_Views_Base_mid_high ,
                                            avg_growth_low, avg_growth_low_mid , avg_growth_mid_high, 
                                            count_vids_low, count_vids_low_mid, count_vids_mid_high))


# add the new columns
real_test = add_column(real_test , .after = "num_digit_chars", num_subscriber = testnum_subscriber)
real_test  = add_column(real_test , .after = "num_digit_chars", num_view =testnum_view)
real_test  = add_column(real_test , .after = "num_digit_chars", avg_growth =testavg_growth)
real_test = add_column(real_test , .after = "num_digit_chars", count_video =testcount_video)


dim(real_test)
View(real_test)



names =  colnames(yt_training)
training = youtube[, names]




# bag.pred = predict(bag.fit, testing, ntree = 500)
pred = predict(random.fit, newdata = real_test)

sample = read.csv("sample.csv")
sample$growth_2_6 = pred

write.csv(sample, file = "Dummy39.csv", row.names = FALSE)








