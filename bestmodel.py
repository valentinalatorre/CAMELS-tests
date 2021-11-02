# Running the best model for Random Forest and Gradient Boosting
# Printing the Features Importance
# Saving results

import numpy as np
import sklearn.metrics
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import optuna

simulation = 'simba' # or 'IllustrisTNG'
path = '/../../'
# - LOADING AND READING train, validation and test data 
# I splitted data using splitting_data.py


in_train = np.loadtxt(path+'train_'+simulation+'.txt', usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13), unpack=True)
out_train = np.loadtxt(path+'train_'+simulation+'.txt', usecols=(14), unpack=True)

in_valid = np.loadtxt(path+'valid_'+simulation+'.txt', usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13), unpack=True)
out_valid = np.loadtxt(path+'valid_'+simulation+'.txt', usecols=(14), unpack=True)

in_test = np.loadtxt(path+'test_'+simulation+'.txt', usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13), unpack=True)
out_test = np.loadtxt(path+'test_'+simulation+'.txt', usecols=(14), unpack=True)

in_train = np.column_stack(in_train)
in_valid = np.column_stack(in_valid)
in_test = np.column_stack(in_test)


print('checking the size of samples')
print(in_train.shape, out_train.shape)
print(in_valid.shape, out_valid.shape)
print(in_test.shape, out_test.shape)

####################################################################
# - RANDOM FOREST
# best parameters
random_state = 42
n_jobs = 1
min_samples_leaf = 2
n_estimators = 197

print('defining model')
rf = RandomForestRegressor(n_estimators = n_estimators, 
                           min_samples_leaf = min_samples_leaf,
                           n_jobs = n_jobs, 
                           random_state=random_state)
print('fitting')
fit = rf.fit(in_train, out_train)


print('making the prediction')
#pred_valid = rf.predict(in_valid)
pred_test = rf.predict(in_test)

results = np.column_stack((pred_test, out_test))
print(results.shape)

fout = 'rf_pred_omegam_'+simulation+'.txt'
np.savetxt(fout, results)
print('IMPORTANCE FEATURES:')
print(fit.feature_importances_)


####################################################################

# - GRADIENT BOOSTING
# best parameters
learning_rate = 0.1307472242538155
max_depth = 14
min_child_weight = 6.267282749175785
gamma = 0.15683653306298598
colsample_bytree = 0.5186649876224394
n_estimators = 102

XGBR = xgb.XGBRegressor(n_estimators=n_estimators, 
										objective='reg:squarederror',
										learning_rate = learning_rate,
										max_depth = max_depth,
										min_child_weight = min_child_weight,
										gamma = gamma,
										colsample_bytree = colsample_bytree)
fit = XGBR.fit(in_train, out_train)

print('IMPORTANCE FEATURES:')
print(fit.feature_importances_)

print('making the prediction')
pred_test = XGBR.predict(in_test)

results = np.column_stack((pred_test, out_test))
print(results.shape)

fout = 'xgb_pred_omegam_'+simulation+'.txt'
np.savetxt(fout, results)#, header=header)
