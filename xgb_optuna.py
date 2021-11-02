# TUNING GRADIENT BOOSTING HYPERPARAMETERS WITH OPTUNA

import numpy as np
import optuna
import xgboost as xgb
from xgboost import XGBRegressor

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

# - GRADIENT BOOSTING

def objective(trial):
	learning_rate = trial.suggest_float('learning_rate', 0.1, 0.6)
	max_depth = trial.suggest_int('max_depth', 4, 15)
	min_child_weight = trial.suggest_float('min_child_weight', 0.5, 7)
	gamma = trial.suggest_float('gamma', 0.05, 0.4, log=True)
	colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1)
	n_estimators = trial.suggest_int('n_estimators', 10,200)

	XGBR = xgb.XGBRegressor(n_estimators=n_estimators, 
											objective='reg:squarederror',
											learning_rate = learning_rate,
											max_depth = max_depth,
											min_child_weight = min_child_weight,
											gamma = gamma,
											colsample_bytree = colsample_bytree)

	fit = XGBR.fit(in_train, out_train)

	print('making the prediction')
	pred_valid = XGBR.predict(in_valid)
	pred_test = XGBR.predict(in_test)

	print('calculating the loss')

	valid_loss = (pred_valid - out_valid)**2
	valid_mse = np.sqrt(np.mean(valid_loss))

	test_loss = (pred_test - out_test)**2
	test_mse = np.sqrt(np.mean(test_loss))

	f = open('loss_XGB_omegam_'+simulation+'.txt', 'a')
	f.write('%d %.3e %.3e \n'%(trial.number, valid_mse, test_mse))
	f.close()
	return valid_mse # = validation error

print('study name')
study_name = 'xgb'
n_trials   = 100 #set to None for infinite
storage    = 'sqlite:///xgb.db'

sampler = optuna.samplers.TPESampler(n_startup_trials=25)
study = optuna.create_study(study_name=study_name, storage=storage, sampler=sampler)
study.optimize(objective, n_trials)

print('Minimum mse: ', study.best_value) #to get the best observed value of the objective function
print('Best parameter: ', str(study.best_params))

trial = study.best_trial
print("Best trial: number {}".format(trial.number))
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
