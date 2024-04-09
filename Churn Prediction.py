# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 18:23:46 2024

@author: lenovo
"""

# =============================================================================
# Introduction
# Find and Visualize the features contributing to churn
# finding best model to predict the churn
# =============================================================================

# Importing necessary libraries
#for Data Review
import pandas as pd
import numpy as np
# For Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. Read and review the Data
# =============================================================================
data = pd.read_csv("C:\\Users\lenovo\Downloads\Churn_Modelling.csv")
uniques = data.nunique()
shapes = data.shape
head = data.head()
types = data.dtypes
nulls = data.isnull().sum()

"""
1.1 The dataset has 14 features and its height is 10000 rows. Beside target value
    there are 5 categorical features.
1.2 Target value is categorical, so it's a categorical problem.
1.3 Lets look at the features one by one
    1.3.01 RowNumber: it's a serial number or index for the data. we can remove 
           as it do not add any value to prediction.
    1.3.02 CustomerId: Id of a customer. we can remove as it do not add any 
           value to prediction.
    1.3.03 Surname: Usually names and surname are not considered as useful feature
           but we can keep it as it defines a clan of a customer.
    1.3.04 CreditScore: Credit score of customer
    1.3.05 Geography: Geographic location of a customer.
    1.3.06 Gender: Gender of a customer
    1.3.07 Age: Age of a customer.
    1.3.08 Tenure: How long a customer relation with the bank
    1.3.09 Balance: Current bank balance of a customer
    1.3.10 NumberOfProducts: number of products that a cuatomer had purchased.
    1.3.11 HasCrCard: whether a customer has a credit card or not.
    1.3.12 isActivemember: whether a customer was/is an active customer.
    1.3.13 EstimatedSalary: Estimated salary by credit.
    1.3.14 Exited: whether a customer exited/ no longer a customer. it's the target
    
1.4 Categorical(Surname, Geography, Gender, NumOfProducts, HasCrCard, IsActiveMember)
    Continous (CreditScore, Tenure, Balance, EstimatedSalary)
    target (Exited)
1.5 Fortunately there are no null values.
~~~~~~~~~~~  Conclusion: remove(RowNumber, CustomerId) and encode (Geography,Gender)~~~~~~~~~
"""


# =============================================================================
# 2. Exploratory Data Analytics
# Exploratory data analytics are the techniques to better understand the data. 
# heat map, kernal density are the examples. we can visualize categorical and continous 
# features against the target value to understand the data.
# =============================================================================

# Visualising data against the target value using matplotlib
#Categorical Data
fig, axarr = plt.subplots(3,2, figsize=(20,12))
sns.countplot(x = 'Surname', hue = 'Exited',data = data,ax =axarr[0][0] )
sns.countplot(x = 'Geography', hue = 'Exited',data = data,ax =axarr[0][1] )
sns.countplot(x = 'Gender', hue = 'Exited',data = data,ax =axarr[1][0] )
sns.countplot(x = 'NumOfProducts', hue = 'Exited',data = data,ax =axarr[1][1] )
sns.countplot(x = 'HasCrCard', hue = 'Exited',data = data,ax =axarr[2][0] )
sns.countplot(x = 'IsActiveMember', hue = 'Exited',data = data,ax =axarr[2][1] )
fig.show()

#Coninous Data
fig1, axarr1 = plt.subplots(2,2, figsize=(20,12))
sns.boxplot(y = 'CreditScore', x = 'Exited', hue='Exited', data = data, ax = axarr1[0][0])
sns.boxplot(y = 'Tenure', x = 'Exited', hue='Exited', data = data, ax = axarr1[0][1])
sns.boxplot(y = 'Balance', x = 'Exited', hue='Exited', data = data, ax = axarr1[1][0])
sns.boxplot(y = 'EstimatedSalary', x = 'Exited', hue='Exited', data = data, ax = axarr1[1][1])
fig1.show()


#correlations between features
fig2, axarr2 = plt.subplots(2,2, figsize=(20,12))
sns.heatmap(data.corr(),annot=True, ax = axarr2[0][0], cmap = 'Greens')
sns.heatmap(data.corr(method='spearman'),annot=True, ax = axarr2[0][1], cmap = 'Blues')
sns.heatmap(data.corr(method= 'kendall'),annot=True, ax = axarr2[1][0], cmap = 'Reds')
fig2.show()


#kernal density estimation
fig3, axarr3 = plt.subplots(2,2, figsize=(20,12))
sns.kdeplot(data = data['CreditScore'], common_norm=False, ax = axarr3[0][0])
sns.kdeplot(data = data['Tenure'], common_norm=False, ax = axarr3[0][1])
sns.kdeplot(data = data['Balance'], common_norm=False, ax = axarr3[1][0])
sns.kdeplot(data = data['EstimatedSalary'], common_norm=False, ax = axarr3[1][1])
fig3.show()

'''
1. Categorical features:
    1.1 surname : as we can see from the bar chart surname is not relevant
    1.2 country : number of customers in germany is less and out of all the countries
        germany banks lose more customers
    1.3 gender : female cusomers churn more
    1.4 number of products: there is not any specific relation visible in number of products
    1.5 has credit card : no visible pattern
    1.6 active member: not active members are churning more
    Conclusion: irrelavent features(surname)
2. continous features:
    all the continous features have nearly same mean and distribution
3. corelations:
    age, balance, number of products, is active member are more impacting the churn
    we can again analyse the corelations after a 
'''


# =============================================================================
# Data preprocessing (MES OF SR)
# =============================================================================

# Mising Values:
"""
from Data review we know that there is no missing values
"""


# Encoding
data1 = pd.get_dummies(data = data, columns=['Geography', 'Gender'])


# Scalling and normalization
''' Scaling is required when we are using methods based on measures how far data
 points are. such as svm or k-nearest neighbors.
 Normalization is more radical transformation in which data is normalized over the
 mean to form a bell curve
 
 '''
from sklearn import preprocessing
# as per kde normalization is not required 
# from data review we know scaling required for NumOfProducts CreditScore Tenure
# Balance EstimatedSalary
continous_valriables = ['CreditScore', 'Tenure', 'Balance', 'EstimatedSalary','NumOfProducts','Age']
scaler = preprocessing.MinMaxScaler()
data1[continous_valriables]= pd.DataFrame(scaler.fit_transform(data1[continous_valriables]))


# Outliers
data1.columns

#1. Variable CreditScore
q1, q3 = np.percentile(data1['CreditScore'], [25,75])
iqr = q3 - q1
Upper_limit = q3 + 1.5*iqr
Lower_limit = q1 - 1.5*iqr
sum(data1.CreditScore > Upper_limit)
sum(data1.CreditScore < Lower_limit)
data1 = data1[(data1.CreditScore <= Upper_limit)]
data1 = data1[(data1.CreditScore >= Lower_limit)]

#2. Variable Age
q1, q3 = np.percentile(data1['Age'], [25,75])
iqr = q3 - q1
Upper_limit = q3 + 1.5*iqr
Lower_limit = q1 - 1.5*iqr
sum(data1.Age > Upper_limit)
sum(data1.Age < Lower_limit)
data1 = data1[(data1.Age <= Upper_limit)]
data1 = data1[(data1.Age >= Lower_limit)]

#3. Variable Tenure
q1, q3 = np.percentile(data1['Tenure'], [25,75])
iqr = q3 - q1
Upper_limit = q3 + 1.5*iqr
Lower_limit = q1 - 1.5*iqr
sum(data1.Tenure > Upper_limit)
sum(data1.Tenure < Lower_limit)
#no outliers in Tenure

#4. Variable Balance
q1, q3 = np.percentile(data1['Balance'], [25,75])
iqr = q3 - q1
Uper_limit = q3 + 1.5*iqr
Lower_limit = q1 - 1.5*iqr
sum(data1.Balance > Uper_limit)
sum(data1.Balance < Lower_limit)
# no outliers in Balance

#4. Variable EstimatedSalary
q1, q3 = np.percentile(data1['EstimatedSalary'], [25,75])
iqr = q3 - q1
Uper_limit = q3 + 1.5*iqr
Lower_limit = q1 - 1.5*iqr
sum(data1.EstimatedSalary > Uper_limit)
sum(data1.EstimatedSalary < Lower_limit)
# no outliers in EstimatedSalary


# Feature Engineering

# 1. we can do feature selection and feature extraction in feature engineering

# 1.1 Here we are performing feature extraction 
# 1.1.1 introducing new feature
data1['BalanceSalaryRatio'] = data1['Balance']/data1['EstimatedSalary']
sns.boxplot(y='BalanceSalaryRatio', x= 'Exited', hue = 'Exited', data = data1)
plt.ylim(-1,4)
"""
We have clearli seen that salary has little impact over the churning. however as seen in 
boxplot balance salary ratio indicates that customers with higher balance salary 
ratio curn more.
"""

# 1.1.2 Introducing new feature
# Given that tenure is a function of age. we introduce a variable aiming to standardize
# tenure over the age
data1['Tenurebyage'] = data1['Tenure']/data1['Age']
sns.boxplot(y = 'Tenurebyage', x = 'Exited', hue = 'Exited', data = data1)
plt.ylim(0, 10)

# 1.1.2 Introducing new feature
data1['CreditscorebyAge'] = data1['CreditScore']/data1['Age']


# Removing unwanted features
data1 = data1.drop(columns = ['RowNumber','CustomerId','Surname'], axis=1)

# Data preparation

# Split Train, test data
df_train = data1.sample(frac=0.8,random_state=200)
df_test = data1.drop(df_train.index)
print(len(df_train))
print(len(df_test))

# Arrange columns by data type for easier manipulation
continuous_vars = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio',
                   'TenureByAge','CreditScoreGivenAge']
cat_vars = ['HasCrCard', 'IsActiveMember','Geography', 'Gender']
df_train = df_train[['Exited'] + continuous_vars + cat_vars]
df_train.head()


'''For the one hot variables, we change 0 to -1 so that the models can capture a negative relation 
where the attribute in inapplicable instead of 0'''
df_train.loc[df_train.HasCrCard == 0, 'HasCrCard'] = -1
df_train.loc[df_train.IsActiveMember == 0, 'IsActiveMember'] = -1
df_train.head()


# One hot encode the categorical variables
lst = ['Geography', 'Gender']
remove = list()
for i in lst:
    if (df_train[i].dtype == np.str or df_train[i].dtype == np.object):
        for j in df_train[i].unique():
            df_train[i+'_'+j] = np.where(df_train[i] == j,1,-1)
        remove.append(i)
df_train = df_train.drop(remove, axis=1)
df_train.head()


# minMax scaling the continuous variables
minVec = df_train[continuous_vars].min().copy()
maxVec = df_train[continuous_vars].max().copy()
df_train[continuous_vars] = (df_train[continuous_vars]-minVec)/(maxVec-minVec)
df_train.head()

# data prep pipeline for test data
def DfPrepPipeline(df_predict,df_train_Cols,minVec,maxVec):
    # Add new features
    df_predict['BalanceSalaryRatio'] = df_predict.Balance/df_predict.EstimatedSalary
    df_predict['TenureByAge'] = df_predict.Tenure/(df_predict.Age - 18)
    df_predict['CreditScoreGivenAge'] = df_predict.CreditScore/(df_predict.Age - 18)
    # Reorder the columns
    continuous_vars = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary','BalanceSalaryRatio',
                   'TenureByAge','CreditScoreGivenAge']
    cat_vars = ['HasCrCard','IsActiveMember',"Geography", "Gender"] 
    df_predict = df_predict[['Exited'] + continuous_vars + cat_vars]
    # Change the 0 in categorical variables to -1
    df_predict.loc[df_predict.HasCrCard == 0, 'HasCrCard'] = -1
    df_predict.loc[df_predict.IsActiveMember == 0, 'IsActiveMember'] = -1
    # One hot encode the categorical variables
    lst = ["Geography", "Gender"]
    remove = list()
    for i in lst:
        for j in df_predict[i].unique():
            df_predict[i+'_'+j] = np.where(df_predict[i] == j,1,-1)
        remove.append(i)
    df_predict = df_predict.drop(remove, axis=1)
    # Ensure that all one hot encoded variables that appear in the train data appear in the subsequent data
    L = list(set(df_train_Cols) - set(df_predict.columns))
    for l in L:
        df_predict[str(l)] = -1        
    # MinMax scaling coontinuous variables based on min and max from the train data
    df_predict[continuous_vars] = (df_predict[continuous_vars]-minVec)/(maxVec-minVec)
    # Ensure that The variables are ordered in the same way as was ordered in the train set
    df_predict = df_predict[df_train_Cols]
    return df_predict

# =============================================================================
# # Model Building
# =============================================================================
# Support functions
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform

# Fit models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Scoring functions
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
# Function to give best model score and parameters
def best_model(model):
    print(model.best_score_)    
    print(model.best_params_)
    print(model.best_estimator_)
def get_auc_scores(y_actual, method,method2):
    auc_score = roc_auc_score(y_actual, method); 
    fpr_df, tpr_df, _ = roc_curve(y_actual, method2); 
    return (auc_score, fpr_df, tpr_df)


# Fit primal logistic regression
param_grid = {'C': [0.1,0.5,1,10,50,100], 'max_iter': [250], 'fit_intercept':[True],'intercept_scaling':[1],
              'penalty':['l2'], 'tol':[0.00001,0.0001,0.000001]}
log_primal_Grid = GridSearchCV(LogisticRegression(solver='lbfgs'),param_grid, cv=10, refit=True, verbose=0)
log_primal_Grid.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
best_model(log_primal_Grid)


# Fit logistic regression with degree 2 polynomial kernel
param_grid = {'C': [0.1,10,50], 'max_iter': [300,500], 'fit_intercept':[True],'intercept_scaling':[1],'penalty':['l2'],
              'tol':[0.0001,0.000001]}
poly2 = PolynomialFeatures(degree=2)
df_train_pol2 = poly2.fit_transform(df_train.loc[:, df_train.columns != 'Exited'])
log_pol2_Grid = GridSearchCV(LogisticRegression(solver = 'liblinear'),param_grid, cv=5, refit=True, verbose=0)
log_pol2_Grid.fit(df_train_pol2,df_train.Exited)
best_model(log_pol2_Grid)


# Fit SVM with RBF Kernel
param_grid = {'C': [0.5,100,150], 'gamma': [0.1,0.01,0.001],'probability':[True],'kernel': ['rbf']}
SVM_grid = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=0)
SVM_grid.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
best_model(SVM_grid)


# Fit SVM with pol kernel
param_grid = {'C': [0.5,1,10,50,100], 'gamma': [0.1,0.01,0.001],'probability':[True],'kernel': ['poly'],'degree':[2,3] }
SVM_grid = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=0)
SVM_grid.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
best_model(SVM_grid)


# Fit random forest classifier
param_grid = {'max_depth': [3, 5, 6, 7, 8], 'max_features': [2,4,6,7,8,9],'n_estimators':[50,100],'min_samples_split': [3, 5, 6, 7]}
RanFor_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, refit=True, verbose=0)
RanFor_grid.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
best_model(RanFor_grid)


# Fit Extreme Gradient boosting classifier
param_grid = {'max_depth': [5,6,7,8], 'gamma': [0.01,0.001,0.001],'min_child_weight':[1,5,10], 'learning_rate': [0.05,0.1, 0.2, 0.3], 'n_estimators':[5,10,20,100]}
xgb_grid = GridSearchCV(XGBClassifier(), param_grid, cv=5, refit=True, verbose=0)
xgb_grid.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
best_model(xgb_grid)


# =============================================================================
# #Fiting Best Model
# =============================================================================
# Fit primal logistic regression
log_primal = LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=250, multi_class='warn',n_jobs=None, 
                                penalty='l2', random_state=None, solver='lbfgs',tol=1e-05, verbose=0, warm_start=False)
log_primal.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)



# Fit logistic regression with pol 2 kernel
poly2 = PolynomialFeatures(degree=2)
df_train_pol2 = poly2.fit_transform(df_train.loc[:, df_train.columns != 'Exited'])
log_pol2 = LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=300, multi_class='warn', n_jobs=None, 
                              penalty='l2', random_state=None, solver='liblinear',tol=0.0001, verbose=0, warm_start=False)
log_pol2.fit(df_train_pol2,df_train.Exited)


# Fit SVM with RBF Kernel
SVM_RBF = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf', max_iter=-1, probability=True, 
              random_state=None, shrinking=True,tol=0.001, verbose=False)
SVM_RBF.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)


# Fit SVM with Pol Kernel
SVM_POL = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,  decision_function_shape='ovr', degree=2, gamma=0.1, kernel='poly',  max_iter=-1,
              probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)
SVM_POL.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)


# Fit Random Forest classifier
RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=8, max_features=6, max_leaf_nodes=None,min_impurity_decrease=0.0,
                            min_impurity_split=None,min_samples_leaf=1, min_samples_split=3,min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
                            oob_score=False, random_state=None, verbose=0,warm_start=False)
RF.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)


# Fit Extreme Gradient Boost Classifier
XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,colsample_bytree=1, gamma=0.01, learning_rate=0.1, max_delta_step=0,max_depth=7,
                    min_child_weight=5, missing=None, n_estimators=20,n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,reg_alpha=0, 
                    reg_lambda=1, scale_pos_weight=1, seed=None, silent=True, subsample=1)
XGB.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)



print(classification_report(df_train.Exited, log_primal.predict(df_train.loc[:, df_train.columns != 'Exited'])))
print(classification_report(df_train.Exited,  log_pol2.predict(df_train_pol2)))
print(classification_report(df_train.Exited,  SVM_RBF.predict(df_train.loc[:, df_train.columns != 'Exited'])))
print(classification_report(df_train.Exited,  SVM_POL.predict(df_train.loc[:, df_train.columns != 'Exited'])))
print(classification_report(df_train.Exited,  RF.predict(df_train.loc[:, df_train.columns != 'Exited'])))
print(classification_report(df_train.Exited,  XGB.predict(df_train.loc[:, df_train.columns != 'Exited'])))



y = df_train.Exited
X = df_train.loc[:, df_train.columns != 'Exited']
X_pol2 = df_train_pol2
auc_log_primal, fpr_log_primal, tpr_log_primal = get_auc_scores(y, log_primal.predict(X),log_primal.predict_proba(X)[:,1])
auc_log_pol2, fpr_log_pol2, tpr_log_pol2 = get_auc_scores(y, log_pol2.predict(X_pol2),log_pol2.predict_proba(X_pol2)[:,1])
auc_SVM_RBF, fpr_SVM_RBF, tpr_SVM_RBF = get_auc_scores(y, SVM_RBF.predict(X),SVM_RBF.predict_proba(X)[:,1])
auc_SVM_POL, fpr_SVM_POL, tpr_SVM_POL = get_auc_scores(y, SVM_POL.predict(X),SVM_POL.predict_proba(X)[:,1])
auc_RF, fpr_RF, tpr_RF = get_auc_scores(y, RF.predict(X),RF.predict_proba(X)[:,1])
auc_XGB, fpr_XGB, tpr_XGB = get_auc_scores(y, XGB.predict(X),XGB.predict_proba(X)[:,1])



plt.figure(figsize = (12,6), linewidth= 1)
plt.plot(fpr_log_primal, tpr_log_primal, label = 'log primal Score: ' + str(round(auc_log_primal, 5)))
plt.plot(fpr_log_pol2, tpr_log_pol2, label = 'log pol2 score: ' + str(round(auc_log_pol2, 5)))
plt.plot(fpr_SVM_RBF, tpr_SVM_RBF, label = 'SVM RBF Score: ' + str(round(auc_SVM_RBF, 5)))
plt.plot(fpr_SVM_POL, tpr_SVM_POL, label = 'SVM POL Score: ' + str(round(auc_SVM_POL, 5)))
plt.plot(fpr_RF, tpr_RF, label = 'RF score: ' + str(round(auc_RF, 5)))
plt.plot(fpr_XGB, tpr_XGB, label = 'XGB score: ' + str(round(auc_XGB, 5)))
plt.plot([0,1], [0,1], 'k--', label = 'Random: 0.5')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')
#plt.savefig('roc_results_ratios.png')
plt.show()

# Make the data transformation for test data
df_test = DfPrepPipeline(df_test,df_train.columns,minVec,maxVec)
df_test = df_test.mask(np.isinf(df_test))
df_test = df_test.dropna()
df_test.shape


print(classification_report(df_test.Exited,  RF.predict(df_test.loc[:, df_test.columns != 'Exited'])))

auc_RF_test, fpr_RF_test, tpr_RF_test = get_auc_scores(df_test.Exited, RF.predict(df_test.loc[:, df_test.columns != 'Exited']),
                                                       RF.predict_proba(df_test.loc[:, df_test.columns != 'Exited'])[:,1])
plt.figure(figsize = (12,6), linewidth= 1)
plt.plot(fpr_RF_test, tpr_RF_test, label = 'RF score: ' + str(round(auc_RF_test, 5)))
plt.plot([0,1], [0,1], 'k--', label = 'Random: 0.5')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')
#plt.savefig('roc_results_ratios.png')
plt.show()