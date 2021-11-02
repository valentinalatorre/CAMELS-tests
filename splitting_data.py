#splitting data in train, validation and test samples.

import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split

path = '/projects/QUIJOTE/CAMELS/cosmo_1gal/'
simulation = 'IllustrisTNG' # or 'SIMBA'

galaxy_properties = np.loadtxt(path+'galaxies_'+simulation+'_z=0.txt')
#print(galaxy_properties.shape)

offset_, length_ = np.loadtxt(path+'offset_'+simulation+'_z=0.txt', unpack=True)
#Need to have offset and length as integers to avoid
#slicing errors when galaxies from each simulation
offset = offset_.astype(int)
length = length_.astype(int)

parameters = np.loadtxt(path+'latin_hypercube_params_'+simulation+'.txt')

#Splitting the data in train, validation, test, based on the simulation. 
#Galaxies in the first 700 simulations go as train data, 
#then the last 300 are equally splotted in test and validation data

length_train = 700 
length_valid = length_train+150
length_test = length_valid+150
train = []
valid = []
test = []
for i in range(length_train):
    galaxies = galaxy_properties[offset[i]:offset[i]+length[i]]
    for j in galaxies:
        train.append(list(j)+list(parameters[i]))

for i in range(length_train, length_valid):
    galaxies = galaxy_properties[offset[i]:offset[i]+length[i]]
    for j in galaxies:
        valid.append(list(j)+list(parameters[i]))
#print(parameters[i])
for i in range(length_valid, length_test):
    galaxies = galaxy_properties[offset[i]:offset[i]+length[i]]
    for j in galaxies:
        test.append(list(j)+list(parameters[i]))
        #print(parameters[i])
        

#Saving data
header = '# | gas mass | stellar mass | black-hole mass | total mass | Vmax | velocity dispersion | gas metallicity | stars metallicity | star-formation rate | spin | peculiar velocity | stellar radius | total radius | Vmax radius | Omega_m | sigma_8 | A_SN1 | A_AGN1 | A_SN2 | A_AGN2 |'

fout_train = path+'train_data.txt'
fout_valid = path+'valid_data.txt'
fout_test = path'test_data.txt'

np.savetxt(fout_train, train, header=header)
np.savetxt(fout_valid, valid, header=header)
np.savetxt(fout_test, test, header=header)
