'''
This script calls the trained C-3PO neural network modules and models the l=m=1 mode periods, using the methology described in Mombarg et al. 2021, 650, A58.
The environment for running this code is stored in 'keras-environment.yml'.

Input files:
C-3PO modules, *.h5
Training set used for the C-3PO NN, *.pkl
The residuals of the NN that are used to assign an uncertainty of the NN per radial order.
Data file(s) with the observed Teff, log_g, log_L, metallicity, f_rot, and the uncertainties on these.
Data file containg the l=m=1 periods to model.

Output files:
File with parameters of the model (mass, age, metallicity, CBM, Dmix, f_rot, Teff, log_g, log, periods, radial orders), the (inverted) covariance matrix, and likelihood (see Mombarg et al.).
File with mass, Xc, and fov, and ranges that can be used to compute a small grid around this solution.

-- Joey Mombarg, 2 June 2022
'''

import numpy as np
import matplotlib
# Depending on your machine, the default backend might be OK.
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import h5py, glob
import pandas as pd
from pathlib import Path
import sys, re, os, subprocess as sp
import matplotlib.cm as cm
import pickle
#from multiprocessing import Pool
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # this is to prevent some random OMP error.
from keras.models import load_model
import sobol_seq
import random
from scipy.spatial import distance
from scipy import stats
from mpl_toolkits import mplot3d

#### --------------------------------------------------------
''' The following parameters need to be set by the user. '''

c3po_DIR         = './C-3PO'
work_DIR         = './Input'
save_DIR         = './Output'
save_ext         = 'c3po-n15-91_Dmix-Z-frot-fix_MD_L-Teff'
KIC              = '12066947'

# Effective temperature.
Teff             = 7330
sigma_Teff       = 70
# Surface gravity.
log_g            = 4.35
sigma_log_g      = 0.30
# metallicity [M/H].
MH               = 0.0
sigma_MH         = 0.1
# Near-core rotation rate (c/d).
f_rot            = 2.1590
sigma_min_f_rot  = 0.0040
sigma_plus_f_rot = 0.0040
# Luminosity log L/Lsun.
log_L            = 0.8118
log_L_err_low    = 0.0058
log_L_err_up     = 0.0059
# Only keep models with n_sig of the uncertainties on Teff, log_g and log_L.
n_sig            = 2
# Number of models sampled over entire parameter space.
N_sample1        = 2000 #5000
# Number of models sampled within consistent mass and Xc ranges.
N_sample2        = 8000 #15000
#### --------------------------------------------------------



### In order to have the exact same result everytime the code is executed, fix the seed.
np.random.seed(691293)

### Set up some stuff for nice-looking plots.
fontsize = 14
color = 'k'
matplotlib.rc('axes',edgecolor= color)
matplotlib.rc('xtick', labelsize = fontsize, color = color)
matplotlib.rc('ytick', labelsize = fontsize, color = color)


c3po_1 = load_model(c3po_DIR + '/C-3PO_Zext_npg15-91_P_0.h5', compile=False)
print('Module 1 loaded')
c3po_2 = load_model(c3po_DIR + '/C-3PO_Zext_npg15-91_P_1.h5', compile=False)
print('Module 2 loaded')
c3po_3 = load_model(c3po_DIR + '/C-3PO_Zext_npg15-91_P_2.h5', compile=False)
print('Module 3 loaded')
c3po_4 = load_model(c3po_DIR + '/C-3PO_Zext_npg15-91_P_3.h5', compile=False)
print('Module 4 loaded')
c3po_5 = load_model(c3po_DIR + '/C-3PO_Zext_npg15-91_P_4.h5', compile=False)
print('Module 5 loaded')
c3po_L = load_model(c3po_DIR + '/C-3PO_spec_0_Zext_log_L.h5', compile=False)
print('Module spectroscopic loaded')
c3po_s = load_model(c3po_DIR + '/C-3PO_spec_0_Zext_log_Teff_g.h5', compile=False)
print('Module spectroscopic loaded')

def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text


''' Load in the training data of the neural network to know the parameter space for the sampling. (Better not to extrapolate.)'''

input_training_set  = pickle.load(open(work_DIR + '/Input_training_set_nodiff_All_Stars_n91-15_Zext.pkl', 'rb'))
output_training_set = pickle.load(open(work_DIR + '/Output_training_set_nodiff_All_Stars_n91-15_Zext.pkl', 'rb'))
input_training_set_spec  = pickle.load(open(work_DIR + '/Input_training_set_nodiff_All_Stars_Zext_logL-Teff-g.pkl', 'rb'))
output_training_set_spec = pickle.load(open(work_DIR + '/Output_training_set_nodiff_All_Stars_Zext_logL-Teff-g.pkl', 'rb'))

### Correction because the radial order labeling was swapped around.
output_training_set_ = {}
keys = list(output_training_set.keys())
K = len(keys)
for i in range(K):
    output_training_set_[np.flip(keys)[i]] = output_training_set[keys[i]]
output_training_set = output_training_set_

Xc = np.array(input_training_set['Xc'])
### The neural nerwork is trained on models that are relaxed, and we omit models in the highly non-linear regime of the hook.
get = np.array([Xc < 0.70]) & np.array([Xc > 0.05])
get = get[0]

set_inputs = []
for i in range(len(input_training_set['M'])):
    entry = []
    if get[i]:
        for key in input_training_set.keys():
            entry.extend([input_training_set[key][i]])
        entry.extend([input_training_set['M'][i] * input_training_set['Xc'][i]])
        entry.extend([input_training_set['M'][i] * input_training_set['Z'][i]])
        set_inputs.append(entry)

set_outputs = []
for i in range(len(output_training_set['20'])):
    entry = []
    if get[i]:
        for key in np.flip([str(n) for n in np.arange(15, 92, 1)]):
            entry.extend([output_training_set[key][i]])
        set_outputs.append(entry)


norm_dict = {}
keys = [k for k in input_training_set.keys()] + ['M*Xc', 'M*Z']

for i in range(len(set_inputs[0])):
    min_val = np.min([e[i] for e in set_inputs])
    max_val = np.max([e[i] for e in set_inputs])
    norm_dict[keys[i]] = [min_val, max_val]


for i in range(len(set_inputs[0])):
    min_val = np.min([e[i] for e in set_inputs])
    max_val = np.max([e[i] for e in set_inputs])
    #print(min_val, max_val)
    for j in range(len(set_inputs)):
        set_inputs[j][i] = (set_inputs[j][i] - min_val) / (max_val - min_val)

### Same for spectroscopic data.
Xc = np.array(input_training_set_spec['Xc'])
get = np.array([Xc < 0.70]) & np.array([Xc > 0.05])
get = get[0]

set_inputs_spec = []
for i in range(len(input_training_set_spec['M'])):
    entry = []
    if get[i]:
        for key in input_training_set_spec.keys():
            entry.extend([input_training_set_spec[key][i]])
        entry.extend([input_training_set_spec['M'][i] * input_training_set_spec['Xc'][i]])
        entry.extend([input_training_set_spec['M'][i] * input_training_set_spec['Z'][i]])
        set_inputs_spec.append(entry)

norm_dict_s = {}
keys_s = [k for k in input_training_set_spec.keys()] + ['M*Xc', 'M*Z']

for i in range(len(set_inputs_spec[0])):
    min_val = np.min([e[i] for e in set_inputs_spec])
    max_val = np.max([e[i] for e in set_inputs_spec])
    norm_dict_s[keys_s[i]] = [min_val, max_val]

def normalize_input_arr(M, Xc, Z, fov, Dmix, omega, norm_dict):
    input = np.array([M, Xc, Z, fov, Dmix, omega, M * Xc, M * Z])
    for i in range(8):
        input[i] = (input[i] - norm_dict[keys[i]][0]) / (norm_dict[keys[i]][1] - norm_dict[keys[i]][0])
    return np.array(input).T

def normalize_input_arr_s(M, Xc, Z, fov, Dmix, norm_dict_s):
    input = np.array([M, Xc, Z, fov, Dmix, M * Xc, M * Z])
    for i in range(len(input)):
        input[i] = (input[i] - norm_dict_s[keys_s[i]][0]) / (norm_dict_s[keys_s[i]][1] - norm_dict_s[keys_s[i]][0])
    return np.array(input).T

def normalize_input(M, Xc, Z, fov, Dmix, norm_dict_s):
    input = [M, Xc, Z, fov, Dmix, M * Xc, M * Z]
    for i in range(len(input)):
        input[i] = (input[i] - norm_dict_s[keys_s[i]][0]) / (norm_dict_s[keys_s[i]][1] - norm_dict_s[keys_s[i]][0])
    return np.array([input])

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def predict_P(M, Xc, Z, fov, Dmix, omega, norm_dict):
    """ Average the predictions of each mode period per radial order over several NN modules with different initial conditions. """
    global c3po_1, c3po_2, c3po_3, c3po_4, c3po_5#, c3po_6
    norm_input = normalize_input_arr(M = M, Xc = Xc, Z = Z, fov = fov, Dmix = Dmix, omega = omega, norm_dict=norm_dict)
    P_pred_1 = c3po_1.predict(norm_input)
    P_pred_2 = c3po_2.predict(norm_input)
    P_pred_3 = c3po_3.predict(norm_input)
    P_pred_4 = c3po_4.predict(norm_input)
    P_pred_5 = c3po_5.predict(norm_input)
    P_pred = np.mean([P_pred_1, P_pred_2, P_pred_3, P_pred_4, P_pred_5], axis = 0)
    dP_pred = (P_pred[:,:-1] - P_pred[:,1:]) * 86400.
    return np.flip(P_pred, axis = 1), np.flip(dP_pred, axis = 1)

def remove_largest_P(sequences):
    if len(sequences) == 1:
        return [sequences[-1][:-1]]
    else:
        if len(sequences[-1]) == 1:
            return sequences[:-1]
        else:
            sequences[-1] = sequences[-1][:-1]
            return sequences

def remove_lowest_P(sequences):
    if len(sequences) == 1:
        return [sequences[0][1:]]
    else:
        if len(sequences[0]) == 1:
            return sequences[1:]
        else:
            sequences[0] = sequences[0][1:]
            return sequences



log_Teff         = np.log10(Teff)
sigma_log_Teff   = 1/(Teff * np.log(10)) * sigma_Teff
Zsun             = 0.0134
Zobs             = 10**(MH) * Zsun
sigma_Zobs       = np.abs(10**(MH+np.log10(Zsun)) * np.log(10.)) * sigma_MH
df = pd.read_csv(work_DIR + f'/Kepler{KIC}_spacings.dat', sep='\t', header = 0, names = ['per', 'e_per', 'ampl', 'e_amp', 'phase', 'e_phase', 'stopcrit'], converters={'per' : strip, 'e_per' : strip, 'ampl' : strip, 'e_amp' : strip, 'phase' : strip, 'e_phase' : strip, 'stopcrit' : strip})

### Load the residuals of the neural network, which are used as a measure for the uncertainties in the merit function.
with open(work_DIR + '/residuals_NN_stds.pkl', 'rb') as handle:
    res = pickle.load(handle)

### Do some restructering of the observations.
P        = []
dP       = []
sigma_P  = []
sigma_dP = []

idx = []

if len(idx) == 2:
	if pat == 0:
		i = 0
	else:
		i = idx[-2] + 1

elif len(idx) == 1:
	i = 0

if KIC in ['2710594', '3448365', '5114382', '6468987', '7583663', '8375138', '9480469', '11907454', '12066947']:
	i = 0  # This star also has observed retrograde modes, so skip those and take the prograde ones instead.

print('period(i), period(i+1), err_period(i), err_period(i+1)')
while df['per'][i+1][0] != '*':
	print(df['per'][i], df['per'][i+1], df['e_per'][i], df['e_per'][i+1])
	if df['per'][i] != '--' and df['per'][i+1] != '--':
		dP_ = (float(df['per'][i+1]) - float(df['per'][i])) * 86400.
		sigma_dP_ = np.sqrt(float(df['e_per'][i+1])**2 + float(df['e_per'][i])**2) * 86400.
		dP.append(dP_)
		P.append(float(df['per'][i]))
		sigma_P.append(float(df['e_per'][i]))
		sigma_dP.append(sigma_dP_)
	else:
		dP.append(np.nan)
		P.append(np.nan)
		sigma_P.append(np.nan)
		sigma_dP.append(np.nan)
	i += 1


P_seqs_obs        = [P[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(P))]
dP_seqs_obs       = [dP[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(dP))]
sigma_P_seqs_obs  = [sigma_P[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(sigma_P))]
sigma_dP_seqs_obs = [sigma_dP[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(sigma_dP))]

P_seqs_obs_sort        = np.copy(P_seqs_obs)
dP_seqs_obs_sort       = np.copy(dP_seqs_obs)
sigma_P_seqs_obs_sort  = np.copy(sigma_P_seqs_obs)
sigma_dP_seqs_obs_sort = np.copy(sigma_dP_seqs_obs)

P_seqs_obs.sort(key =len, reverse = True)
dP_seqs_obs.sort(key =len, reverse = True)
sigma_P_seqs_obs.sort(key =len, reverse = True)
sigma_dP_seqs_obs.sort(key =len, reverse = True)

P_min_obs = np.nanmin([np.nanmin(P_seqs_obs[i][0]) for i in range(len(P_seqs_obs))])
P_max_obs = np.nanmax([np.nanmax(P_seqs_obs[i][-1]) for i in range(len(P_seqs_obs))])

P_seqs        = np.copy(P_seqs_obs_sort)
dP_seqs       = np.copy(dP_seqs_obs_sort)
sigma_P_seqs  = np.copy(sigma_P_seqs_obs_sort)
sigma_dP_seqs = np.copy(sigma_dP_seqs_obs_sort)

def run_c3po(work_DIR, save_DIR, save_ext, KIC, Teff, sigma_Teff, log_g, sigma_log_g, MH, sigma_MH, f_rot,
sigma_min_f_rot, sigma_plus_f_rot, log_L, log_L_err_low, log_L_err_up, n_sig, N_sample1, N_sample2):

    global P_seqs, dP_seqs, sigma_P_seqs, sigma_dP_seqs

    P_closest = []
    dP_closest = []
    n_pg_closest = []
    theta1 = {'M' : [], 'Xc' : [], 'Z' : [], 'fov' : [], 'Dmix' : [], 'omega' : []}
    P_pred_arr   = []
    dP_pred_arr  = []
    log_Teff_arr = []
    log_g_arr    = []
    log_L_arr     = []
    MI = [N_sample1, N_sample2] # Number of models sampled by NN during each iteration.
    chi2 = []
    MD   = []
    Y_arr = []

    ### During the first iteration, the whole parameter space is sampled. The next second one, a denser sampling of the n_sig in L, Teff, log_g (or some of these) is done.
    for iter in range(2):
        if iter == 0:
            mi = MI[iter]
            for m in range(mi):
                rand_num = [random.random() for i in range(6)]
                M    = np.round(rand_num[0] * (norm_dict['M'][1] - norm_dict['M'][0]) + norm_dict['M'][0], 3)
                Xc   = np.round(rand_num[1] * (norm_dict['Xc'][1] - norm_dict['Xc'][0]) + norm_dict['Xc'][0], 3)
                fov  = np.round(rand_num[3] * (norm_dict['fov'][1] - norm_dict['fov'][0]) + norm_dict['fov'][0], 4)
                ### Dmix fix at 1 cm^/s, but varying this parameter is also possible.
                Dmix = 1.0 #np.round(rand_num[4] * (norm_dict['Dmix'][1] - norm_dict['Dmix'][0]) + norm_dict['Dmix'][0], 1)
                rand_sign = random.choice([-1, 1])
                if rand_sign == -1:
                    omega = f_rot - rand_num[5]*sigma_min_f_rot
                elif rand_sign == 1:
                    omega = f_rot + rand_num[5]*sigma_plus_f_rot

                Z = np.max([0.011, np.min([0.023, np.round(Zobs + rand_num[2] * 2 * sigma_Zobs, 3)])])
                theta1['M'].append(M)
                theta1['Xc'].append(Xc)
                theta1['Z'].append(Z)
                theta1['fov'].append(fov)
                theta1['Dmix'].append(Dmix)
                theta1['omega'].append(omega)

        if iter == 1:
            mi = MI[iter]
            M_max  = np.max(np.array(theta1['M'])[get])
            M_min  = np.min(np.array(theta1['M'])[get])
            Xc_max = np.max(np.array(theta1['Xc'])[get])
            Xc_min = np.min(np.array(theta1['Xc'])[get])
            for m in range(mi):
                rand_num = [random.random() for i in range(6)]
                M    = np.round(rand_num[0] * (M_max - M_min) + M_min, 3)
                Xc   = np.round(rand_num[1] * (Xc_max - Xc_min) + Xc_min, 3)
                fov  = np.round(rand_num[3] * (norm_dict['fov'][1] - norm_dict['fov'][0]) + norm_dict['fov'][0], 4)
                Dmix = 1.0 #np.round(rand_num[4] * (norm_dict['Dmix'][1] - norm_dict['Dmix'][0]) + norm_dict['Dmix'][0], 1)
                rand_sign = random.choice([-1, 1])
                if rand_sign == -1:
                    omega = f_rot - rand_num[5]*sigma_min_f_rot
                elif rand_sign == 1:
                    omega = f_rot + rand_num[5]*sigma_plus_f_rot
                Z = np.max([0.011, np.min([0.023, np.round(Zobs + rand_num[2] * 2 * sigma_Zobs, 3)])]) # Do not exceed the limits of Z in the training data.
                theta1['M'].append(M)
                theta1['Xc'].append(Xc)
                theta1['Z'].append(Z)
                theta1['fov'].append(fov)
                theta1['Dmix'].append(Dmix)
                theta1['omega'].append(omega)
        print(f'Parameters for model KIC{KIC} generated, iteration', iter)
        P_pred_arr_, dP_pred_arr_ = predict_P(M = np.array(theta1['M'][ iter*MI[0]:MI[iter]+iter*MI[0]]), Xc = np.array(theta1['Xc'][ iter*MI[0]:MI[iter]+iter*MI[0]]), Z = np.array(theta1['Z'][ iter*MI[0]:MI[iter]+iter*MI[0]]), fov = np.array(theta1['fov'][ iter*MI[0]:MI[iter]+iter*MI[0]]), Dmix = np.array(theta1['Dmix'][ iter*MI[0]:MI[iter]+iter*MI[0]]), omega = np.array(theta1['omega'][ iter*MI[0]:MI[iter]+iter*MI[0]]), norm_dict=norm_dict)
        P_pred_arr.extend(P_pred_arr_)
        dP_pred_arr.extend(dP_pred_arr_)
        norm_input_s = normalize_input_arr_s(M = np.array(theta1['M'][ iter*MI[0]:MI[iter]+iter*MI[0]]), Xc = np.array(theta1['Xc'][ iter*MI[0]:MI[iter]+iter*MI[0]]), Z = np.array(theta1['Z'][ iter*MI[0]:MI[iter]+iter*MI[0]]), fov = np.array(theta1['fov'][ iter*MI[0]:MI[iter]+iter*MI[0]]), Dmix = np.array(theta1['Dmix'][ iter*MI[0]:MI[iter]+iter*MI[0]]), norm_dict_s=norm_dict_s)
        S_pred_arr = c3po_s.predict(norm_input_s)
        log_Teff_arr.extend([s[0] for s in S_pred_arr])
        log_g_arr.extend([s[1] for s in S_pred_arr])
        L_pred_arr   = c3po_L.predict(norm_input_s)
        log_L_arr.extend([s[0] for s in L_pred_arr])


        for p in range(len(P_pred_arr)):
            Y_arr.append(list(P_pred_arr[p]))

        Vtot_P = np.cov(Y_arr, rowvar = False)


        while 1:
            print('New iteration')
            for P_pred, dP_pred in zip(P_pred_arr_, dP_pred_arr_):
                P_closest_sub  = []
                dP_closest_sub = []
                n_pg_closest_sub = []
                for s in P_seqs: #np.delete(P_seqs, 3):
                    P_star = np.min(s) #s[0]
                    ii = np.argmin(np.abs(P_pred - P_star))
                    n_pg = np.arange(15, 92, 1)
                    picked_n = [val for sublist in n_pg_closest_sub for val in sublist]

                    ### Throw away the model if one of the following conditions are met:
                    ### 1) The chosen radial order has already been assigned to the first sequence.
                    ### 2) Radial orders lower or higher than those computed are selected.
                    ### 3) If there are gaps in the observed radial orders, this should also be reflected in the selection of the radial orders. E.g. if 20, 21 are already picked for one sequence, the adjecent sequences should not contain 19 or 22.

                    if (n_pg[ii] in picked_n)  or (n_pg[ii]-1 in picked_n) or (n_pg[ii]+1 in picked_n) or (len(n_pg[ii: ii+len(s)]) != len(s)) or (len(P_pred[ii: ii+len(s)]) != len(P_pred[ii: ii+len(s)+1][1:] - P_pred[ii: ii+len(s)+1][:-1])): # or (n_pg[min(len(n_pg)-1, np.max(ii+len(s)))] in picked_n) or (ii > len(n_pg)-1)
                        #print('NO MATCH FOUND!', (n_pg[ii] in picked_n) , (len(n_pg[ii: ii+len(s)]) != len(s)))
                        emp_arr = np.empty(len(s))
                        emp_arr[:] = np.nan
                        P_closest_sub.append(emp_arr)
                        dP_closest_sub.append(emp_arr)
                        n_pg_closest_sub.append(emp_arr)
                    else:
                        P_closest_sub.append(P_pred[ii: ii+len(s)])
                        dP_closest_sub.append((P_pred[ii: ii+len(s)+1][1:] - P_pred[ii: ii+len(s)+1][:-1]) * 86400.)
                        n_pg_closest_sub.append(n_pg[ii: ii+len(s)])
                P_closest.append(P_closest_sub)
                dP_closest.append(dP_closest_sub)
                n_pg_closest.append(n_pg_closest_sub)


            Y = []
            for i in range(len(P_closest)):
                Y.append(list([val for sublist in P_closest[i] for val in sublist])  + list([val for sublist in dP_closest[i] for val in sublist])) #list(P_closest) + list(dP_closest)

            Yobs = np.array(list([val for sublist in P_seqs for val in sublist]) + list([val for sublist in dP_seqs for val in sublist]))

            h = int(0.5 * len(Y[0]))
            Y = np.matrix(Y)
            Yobs = np.matrix(list(Yobs))

            Lambda = np.diag(list([val**2 for sublist in sigma_P_seqs for val in sublist]) + list([(val)**2 for sublist in sigma_dP_seqs for val in sublist]))

            ### Add uncertainty from NN.
            MD   = []
            L    = []
            chi2 = []
            Vinv = []
            V    = []
            for i in range(len(Y)):
                sigma_P_NN  = []
                sigma_dP_NN = []
                for ss in range(len(P_closest[i])):
                    sigma_P_NN_  = []
                    sigma_dP_NN_ = []
                    for nn in n_pg_closest[i][ss]:
                        if np.isnan(nn):
                            sigma_P_NN_.append(np.nan)
                            sigma_dP_NN_.append(np.nan)
                        else:
                            sigma_P_NN_.append(res[str(nn)])
                            sigma_dP_NN_.append(np.sqrt(res[str(nn)]**2 + res[str(nn+1)]**2) * 86400.)
                    sigma_P_NN.append(sigma_P_NN_)
                    sigma_dP_NN.append(sigma_dP_NN_)
                Sigma = np.diag(list([val**2 for sublist in sigma_P_NN for val in sublist]) + list([(val)**2 for sublist in sigma_dP_NN for val in sublist]))
                Sigma_P  = np.diag(list([val**2 for sublist in sigma_P_NN for val in sublist]))
                Sigma_dP = np.diag(list([(val)**2 for sublist in sigma_dP_NN for val in sublist]))
                Lambda_inv = np.linalg.inv(Lambda + Sigma)
                chi2.append(float(np.matmul((Y[i] - Yobs[0]), np.matmul(Lambda_inv, (Y[i] - Yobs[0]).T))))

                n_idx = [n_ - 15 for sublist in n_pg_closest[i] for n_ in sublist]
                Vcut_P  = Vtot_P

                if np.sum(np.isnan(n_idx)) == 0:
                    nc = []
                    for n in range(77):
                        if n not in n_idx:
                            nc.extend([n])

                    Vcut_P  = np.delete(Vcut_P,  (nc), axis =0)
                    Vcut_P  = np.delete(Vcut_P,  (nc), axis =1)

                    Vinv_ = np.linalg.inv(Vcut_P + Sigma_P)
                    MDi = float(np.matmul((Y[i,0:h] - Yobs[0,0:h]), np.matmul(Vinv_, (Y[i,0:h] - Yobs[0,0:h]).T)))
                    ln_detV = np.log(np.linalg.det(Vcut_P + Sigma_P))
                    L.append(np.exp(-0.5*(ln_detV + MDi)))
                    MD.append(MDi)

                    Vinv.append(Vinv_)
                    V.append(Vcut_P + Sigma_P)
                else:
                    MD.append(np.nan)
                    Vinv.append(np.zeros((h,h)))
                    V.append(np.zeros((h,h)))
                    L.append(np.nan)



            '''
            Only models that are consistent with the observerd log L, log Teff, log g within n_sig are considered.
            '''
            get = np.array(log_Teff_arr < log_Teff + n_sig * sigma_log_Teff) & np.array(log_Teff_arr > log_Teff - n_sig * sigma_log_Teff) & np.array(np.array(log_g_arr) < log_g +n_sig * sigma_log_g) & np.array(np.array(log_g_arr) > log_g - n_sig * sigma_log_g) & np.array(np.array(log_L_arr) < log_L + n_sig * log_L_err_up) & np.array(np.array(log_L_arr) > log_L - n_sig * log_L_err_low)
            #get = np.array(log_Teff_arr < log_Teff + n_sig * sigma_log_Teff) & np.array(log_Teff_arr > log_Teff - n_sig * sigma_log_Teff) & np.array(np.array(log_L_arr) < log_L + n_sig * log_L_err_up) & np.array(np.array(log_L_arr) > log_L - n_sig * log_L_err_low)
            #get = np.array(log_Teff_arr < log_Teff + n_sig * sigma_log_Teff) & np.array(log_Teff_arr > log_Teff - n_sig * sigma_log_Teff) & np.array(log_g_arr < log_g +n_sig * sigma_log_g) & np.array(log_g_arr > log_g - n_sig * sigma_log_g)
            #get = np.array(np.array(log_L_arr) < log_L + n_sig * log_L_err_up) & np.array(np.array(log_L_arr) > log_L - n_sig * log_L_err_low)
            if np.sum(get) == 0:
                raise Exception("No models found that are consistent with the observed atmspheric observables.")

            get_ = get[iter*MI[0]:MI[iter]+iter*MI[0]]


            merit = -1*np.array(L) # Maximizing L = minimizing -L
            merit /= np.nanmax(L)

            chi2_sel = np.array(merit[iter*MI[0]:MI[iter]+iter*MI[0]])[get_]
            MD_sel   = np.array(MD[iter*MI[0]:MI[iter]+iter*MI[0]])[get_]

            ### If some periods can never be matched to a radial order (i.e. outside the radial order range) with the spectroscopic error box, remove the largest period and try again.

            P_min_get = np.nanmin(P_pred_arr_[get_], axis = 1) #np.array([np.nanmin(P_closest[i][0]) for i in range(len(P_closest))])[get]
            P_max_get = np.nanmax(P_pred_arr_[get_], axis = 1) #np.array([np.nanmax(P_closest[i][-1]) for i in range(len(P_closest))])[get]
            min_fail = np.sum([P_min_get > P_min_obs])
            max_fail = np.sum([P_max_get < P_max_obs])
            if np.sum(np.isnan(chi2_sel)) > 0.1 * len(chi2_sel) and (min_fail > max_fail) and iter == 0: #0.1
                print(P_seqs)
                P_seqs        = remove_lowest_P(P_seqs)
                print(P_seqs)
                dP_seqs       = remove_lowest_P(dP_seqs)
                sigma_P_seqs  = remove_lowest_P(sigma_P_seqs)
                sigma_dP_seqs = remove_lowest_P(sigma_dP_seqs)
                chi2 = []
                MD   = []
                L    = []
                Vinv = []
                V    = []
                P_closest    = []
                dP_closest   = []
                n_pg_closest = []
            elif np.sum(np.isnan(chi2_sel)) > 0.1 * len(chi2_sel) and (min_fail < max_fail) and iter == 0:
                print(P_seqs)
                P_seqs        = remove_largest_P(P_seqs)
                print(P_seqs)
                dP_seqs       = remove_largest_P(dP_seqs)
                sigma_P_seqs  = remove_largest_P(sigma_P_seqs)
                sigma_dP_seqs = remove_largest_P(sigma_dP_seqs)
                chi2 = []
                MD   = []
                L    = []
                Vinv = []
                V    = []
                P_closest    = []
                dP_closest   = []
                n_pg_closest = []
            else:
                break

    idx_sort = np.argsort(np.array(merit)[get])

    f1, ax = plt.subplots(figsize = (6,5))
    f1.subplots_adjust(left = 0.15, right = 0.95, bottom = 0.15, top = 0.95)
    log_merit     = (np.abs(np.array(merit)))
    log_merit_get = (np.abs(np.array(merit))[get])
    zz = ax.scatter(x = np.array(theta1['M']), y= np.array(theta1['Xc']), marker = 'o', c = log_merit, cmap='viridis_r', s=2, vmin = np.nanmin(log_merit), vmax = np.nanmax(log_merit), alpha=0.35)
    ax.scatter(x = np.array(theta1['M'])[get], y= np.array(theta1['Xc'])[get], marker = 'o', color = 'k', s=3)#, edgecolor ='k')
    zz2 = ax.scatter(x = np.array(theta1['M'])[get], y= np.array(theta1['Xc'])[get], marker = 'o', c = log_merit_get, cmap='viridis_r', s=2, vmin = np.nanmin(log_merit), vmax = np.nanmax(log_merit))#, edgecolor ='k')
    ax.plot(np.array(theta1['M'])[np.nanargmin(merit)], np.array(theta1['Xc'])[np.nanargmin(merit)], 'o', color = 'k', markersize = 10, markeredgecolor = 'k')
    ax.plot(np.array(theta1['M'])[get][idx_sort][0], np.array(theta1['Xc'])[get][idx_sort][0], marker = '*', color = 'r', markersize = 20, markeredgecolor = 'k')
    cb = plt.colorbar(zz2)# r'$\log\,{\rm MD}$')
    cb.set_label(r'$\mathcal{L}_{\rm norm}$', fontsize = fontsize)
    ax.set_xlabel(r'$M_\star\,[{\rm M_\odot}]$', fontsize = fontsize)
    ax.set_ylabel(r'$X_{\rm c}$', fontsize = fontsize)
    ax.set_xlim(1.3, 2.0)
    ax.set_ylim(0.05, 0.7)
    f1.savefig(save_DIR + f'/KIC{KIC}_M-Xc_chi2_logTgL-sig{n_sig}_{save_ext}.png', dpi = 200)
    plt.show(block = False)

    for key in theta1.keys():
        print(key, np.array(theta1[key])[get][idx_sort][0])

    merit = np.array(merit)
    n_pg_bm  = np.array(n_pg_closest)[get][np.nanargmin(merit[get])]
    P_bm  = np.array(P_closest)[get][np.nanargmin(merit[get])]
    dP_bm = np.array(dP_closest)[get][np.nanargmin(merit[get])]
    f2, ax = plt.subplots(figsize = (8,3))
    f2.subplots_adjust(left = 0.15, right = 0.95, bottom = 0.20, top = 0.85)
    label_set = False
    for pobs, dpobs in zip(P_seqs_obs, dP_seqs_obs):
        if not label_set:
            ax.plot(pobs, dpobs, '-o', color = 'k', label = 'Observations')
            label_set = True
        else:
            ax.plot(pobs, dpobs, '-o', color = 'k')
        connect_style = '--o'

    label_set = False
    for pbm, dpbm in zip(P_bm, dP_bm):
        if not label_set:
            ax.plot(pbm, dpbm, '-o', color = 'r', label = 'Best model NN')
            label_set = True
        else:
            ax.plot(pbm, dpbm, '-o', color = 'r')
    P_bm   =np.array([val for sublist in P_bm for val in sublist])
    dP_bm  =np.array([val for sublist in dP_bm for val in sublist])
    P_obs  =np.array([val for sublist in P_seqs for val in sublist])
    dP_obs =np.array([val for sublist in dP_seqs for val in sublist])
    ax.plot(P_bm[np.argsort(P_bm)], dP_bm[np.argsort(P_bm)], connect_style, color = 'r')
    ax.plot(P_obs[np.argsort(P_obs)], dP_obs[np.argsort(P_obs)], connect_style, color = 'k')
    ax.set_xlabel(r'$P\,[{\rm d}]$', fontsize = fontsize)
    ax.set_ylabel(r'$\Delta P\,[{\rm s}]$', fontsize = fontsize)
    ax.legend(frameon = False, fontsize = fontsize)
    M_bm    = np.array(theta1['M'])[get][idx_sort][0]
    Xc_bm   = np.array(theta1['Xc'])[get][idx_sort][0]
    Z_bm    = np.array(theta1['Z'])[get][idx_sort][0]
    Dmix_bm = np.array(theta1['Dmix'])[get][idx_sort][0]
    fov_bm  = np.array(theta1['fov'])[get][idx_sort][0]
    frot_bm = np.round(np.array(theta1['omega'])[get][idx_sort][0], 4)
    ax.set_title(fr'${M_bm}\,{{\rm M_\odot}}, {{X_{{\rm c}}}} = {Xc_bm}, {{Z}} = {Z_bm}, {{D_{{\rm mix}}}} = {Dmix_bm}, {{f_{{\rm ov}}}} = {fov_bm}, {{f_{{\rm rot}}}} = {frot_bm}{{\rm d^{{-1}}}}$')
    f2.savefig(save_DIR + f'/KIC{KIC}_P-dP_best_model_logTgL-sig{n_sig}_{save_ext}.png', dpi = 200)
    plt.show(block = False)

    theta1['log_Teff'] = log_Teff_arr
    theta1['log_g']    = log_g_arr
    theta1['log_L']    = log_L_arr
    theta1['P']        = P_closest
    theta1['n_pg']     = n_pg_closest
    theta1['dP']       = dP_closest
    theta1['chi2']     = chi2
    theta1['MD']       = MD
    theta1['Vinv']     = Vinv
    theta1['V']        = V
    theta1['L(MD)']    = L

    with open(save_DIR + f'/Fit_KIC{KIC}_{save_ext}.pkl', 'wb') as handle:
        pickle.dump(theta1, handle)

    M_get   = np.array(theta1['M'])[get]
    Xc_get  = np.array(theta1['Xc'])[get]
    fov_get = np.array(theta1['fov'])[get]
    L_get   = np.array(L)[get]
    P_D = 0
    Q = len(M_get)
    for ip in range(Q):
        if not np.isnan(L_get[ip]):
            P_D += L_get[ip]*(list(M_get).count(M_get[ip])/Q) * (list(Xc_get).count(Xc_get[ip])/Q) * (list(fov_get).count(fov_get[ip])/Q)


    P_err = 0
    L_get[np.isnan(L_get)] = 0
    sidx     = np.flip(np.argsort(L_get))
    L_sort   = L_get[sidx]
    M_sort   = M_get[sidx]
    Xc_sort  = Xc_get[sidx]
    fov_sort = fov_get[sidx]
    lidx  = 0
    ### Sum the list of models sorted by increasing likelihood until 68% is reached.
    while P_err < 0.68 * P_D:
        P_err += L_sort[lidx] *(list(M_sort).count(M_sort[lidx])/Q) * (list(Xc_sort).count(Xc_sort[lidx])/Q) * (list(fov_sort).count(fov_sort[lidx])/Q)
        lidx += 1


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(M_sort[0:lidx], Xc_sort[0:lidx], fov_sort[0:lidx], 'o', color = 'grey')
    ax.set_xlabel(r'$M_\star\,[{\rm M_\odot}]$', fontsize = fontsize, labelpad = 10)
    ax.set_ylabel(r'$X_{\rm c}$', fontsize = fontsize, labelpad = 10)
    ax.set_zlabel(r'$f_{\rm ov}$', fontsize = fontsize, labelpad = 10)

    theta_lim2 = {'M' : {'min' : np.min(M_sort[0:lidx]), 'opt' : M_bm, 'max' : np.max(M_sort[0:lidx])}, 'Xc' : {'min' : np.min(Xc_sort[0:lidx]), 'opt' : Xc_bm, 'max' : np.max(Xc_sort[0:lidx])}, 'fov' : {'min' : np.min(fov_sort[0:lidx]), 'opt' : fov_bm, 'max' : np.max(fov_sort[0:lidx])}}

    n_pg_bm_ = [n_ for sublist in n_pg_bm for n_ in sublist]
    sigma_P_seqs_ = [n_ for sublist in sigma_P_seqs for n_ in sublist]
    std = [res[str(nn)] for nn in n_pg_bm_]

    n_pg_get = np.array(n_pg_closest)[get]
    P_closest_get = np.array(P_closest)[get]
    Vinv_get  = np.array(Vinv)[get]
    V_get     = np.array(V)[get]
    theta_mc = {'M' : [], 'Xc' : [], 'fov' : []}

    print('Parameter', 'MLE      ,', 'Lower error,', 'Upper error')
    print('M        ', theta_lim2['M']['opt'], np.round(theta_lim2['M']['opt'] - theta_lim2['M']['min'], 3), np.round(theta_lim2['M']['max'] - theta_lim2['M']['opt'], 3))
    print('Xc       ', theta_lim2['Xc']['opt'], np.round(theta_lim2['Xc']['opt'] - theta_lim2['Xc']['min'], 3), np.round(theta_lim2['Xc']['max'] - theta_lim2['Xc']['opt'], 3))
    print('fov      ', theta_lim2['fov']['opt'], np.round(theta_lim2['fov']['opt'] - theta_lim2['fov']['min'], 4), np.round(theta_lim2['fov']['max'] - theta_lim2['fov']['opt'], 4))

    plt.show(block = False)

    with open(save_DIR + f'/Best_model_KIC{KIC}_{save_ext}.pkl', 'wb') as handle:
        pickle.dump(theta_lim2, handle)
    print('-----------------------------------------')

run_c3po(work_DIR, save_DIR, save_ext, KIC, Teff, sigma_Teff, log_g, sigma_log_g, MH, sigma_MH, f_rot, sigma_min_f_rot, sigma_plus_f_rot, log_L, log_L_err_low, log_L_err_up, n_sig, N_sample1, N_sample2)
