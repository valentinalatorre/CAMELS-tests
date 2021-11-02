# TUNING RANDOM FOREST HYPERPARAMETERS WITH OPTUNA

import numpy as np
import sklearn.metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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


# - RANDOM FOREST

random_state = 42
n_jobs = 1
def objective(trial):

    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 100) 
    n_estimators = trial.suggest_int('n_estimators', 10,200)


    rf = RandomForestRegressor(n_estimators = n_estimators, 
                               min_samples_leaf = min_samples_leaf,
                               n_jobs = n_jobs, 
                               random_state=random_state)

    fit = rf.fit(in_train, out_train)


    print('making the prediction')
    pred_valid = rf.predict(in_valid)
    pred_test = rf.predict(in_test)
    
    print('calculating the loss')
    
    valid_loss = (pred_valid - out_valid)**2
    valid_mse = np.sqrt(np.mean(valid_loss))
    
    test_loss = (pred_test - out_test)**2
    test_mse = np.sqrt(np.mean(test_loss))

    f = open('loss_rf_omegam_'+simulation+'.txt', 'a')
    f.write('%d %.3e %.3e \n'%(trial.number, valid_mse, test_mse))
    f.close()
    return valid_mse 

study_name = 'RF'
n_trials   = 100 #set to None for infinite
storage    = 'sqlite:///rf.db'

sampler = optuna.samplers.TPESampler(n_startup_trials=25)
study = optuna.create_study(study_name=study_name, storage=storage, sampler=sampler)
study.optimize(objective, n_trials)

#print('Minimum mse: ', + str(study.best_value))
#print('Best parameter: ', + str(study.best_params))

trial = study.best_trial
print("Best trial: number {}".format(trial.number))
print("  Minimum mse: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

