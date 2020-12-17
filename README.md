# Predicting_Youtube_Video_Views_Growth
In this project, we aim to predict the percentage change in views on a video between the second and sixth hour since its publishing.

## Preprocessing
First by looking at the structure of the data set, we deleted variables that have zero standard devia-
tion(variables that have the same unique value) because it will have no meaning for our analysis. Before
going any further, we identied all of the correlated variables and we want to remove them since it may
cause problems in the future modelling and prediction. We also decided to convert the published date of the
observation into hour only. This is not only because we believe it will have an impact on the views between
the second and the sixth hour since its publishing, also we'll have one more perspective to understand our
problem. We also found that some variables like num subscriber low mid and etc are repetitive, and we see
that from the feature description le, num subscriber,num view,avg growth,and count vids are all binary
variables with low,mid,and high levels as 0 or 1. So we decided to convert them to categorical variable in
the method in table below. So all 3 num subscriber will become on categorical variable, and the other 3 are
the same. This is easier for the model later because we shrunk 3 variables into only 1 but it still represent
the same meaning. Example in Table 1 as how we convert 3 variables into 1.


## Feature Selection(Shrinkage Methods: LASSO, RIDGE, Elastic Net)
There are over 250 features in the original data set and even with preprocessing and removing highly correlated variables, we still ended up having too many features to have a good interpretability. It is also computational heavy when we have too many features and most importantly the “Curse of Dimensionality” which leads to a bad prediction. Hence, we utilized shrinkage methods to shrink trivial features’ coefficient toward zero, in other words, we don’t want those variables that have zero coefficient in our final model.\\
\\
LASSO, RIDGE, Elastic Net: First by splitting 80\% of our data into training set and 20\% as our validation set. After standardizing the data set, we can decide LASSO, Ridge, or Elastic Net by fitting a 10-fold cross-validation on our training data set with different alpha from 0, 0.1, 0.2,...1 (alpha = 0 corresponds to Ridge, alpha = 1 corresponds to LASSO and alpha between 0 and 1 corresponds to Elastic Net) to decide our parameter alpha. We'll never know which method gives the best result, so comparing all methods' performance is necessary here.
