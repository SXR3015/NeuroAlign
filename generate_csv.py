import csv
import numpy as np
import os
from math import ceil
from utils import OsJoin
from sklearn.model_selection import KFold
from opts import parse_opts
import pandas as pd
import fnmatch
opt = parse_opts()
root = opt.data_root_path

#health_dir = opt.data_type + '_healthy'
#MCI_dir = opt.data_type + '_MCI'
csv_save_dir = OsJoin('csv/', opt.data_type, opt.category)

test_ratio = 0.1
n_fold = opt.n_fold
data_health = []
label_health = []
data_MCI = []
label_MCI = []
data_SMC = []
label_SMC = []
Subs=list()
least_subs_name=r'D:\sxr_bak\features_adni\dfc'
for filename in os.listdir(least_subs_name):
    sub = filename.split('dfc_Covswra_')[1].split('_rsfMRI_timeseries_Dosenbach164.mat')[0]
    # sub_name_delPRE = sub.split('_PRE')[0]
    Subs.append(sub)
# sub_name = os.listdir(least_subs_name)
fea_num = len(os.listdir(root))
fea_num = 4
# sub_num=len(Subs)
# data_health=np.empty([sub_num, fea_i])
# data_MCI=np.empty([sub_num, fea_i])
# data_SMC=np.empty([sub_num, fea_i])
# sub_n=0
HC_num = 0
MCI_num = 0
SMC_num = 0
sub_info_excel = pd.read_csv(r'E:\sxr\APP\edgedata\184\ADNI2_ADNI3.csv', header=0, usecols=[0, 5])

for sub in Subs:
    for feature in os.listdir(root):
        if 'csv' in feature:
            csv_contain=True
            continue
        else:
            csv_contain = False
        for sub_fea in os.listdir(os.path.join(root, feature)):
#or str(sub_name_delPRE)+'_PRE' in str(sub_fea)
            # if sub == '053_S_0919':
            #     print('stop')
            if str(sub) in str(sub_fea):
                sub_index = list(sub_info_excel['Subject ID']).index(sub)
                sub_category = sub_info_excel['DX Group'][sub_index]
                if 'Normal' in sub_category and 'CN' in opt.category:
                    data_health.append(OsJoin(root, feature, sub_fea))
                    if opt.n_classes == 3:
                        label_health.append(0)
                    elif opt.n_classes == 2:
                        label_health.append(0)
                    HC_num = HC_num+1
                elif 'MCI' in sub_category and 'MCI' in opt.category:
                    data_MCI.append(OsJoin(root, feature, sub_fea))
                    if opt.n_classes == 2 and 'CN' in opt.category:
                        label_MCI.append(1)
                    else :
                        label_MCI.append(1)
                    MCI_num = MCI_num+1
                elif 'SMC' in sub_category and 'SMC' in opt.category:
                    data_SMC.append(OsJoin(root, feature, sub_fea))
                    SMC_num = SMC_num+1
                    if opt.n_classes == 3:
                      label_SMC.append(2)
                    elif opt.n_classes == 2 and 'CN_MCI_SMC' in opt.category:
                      label_SMC.append(1)
                    elif opt.n_classes == 2 and 'CN_SMC' in opt.category:
                      label_SMC.append(1)
                    elif opt.n_classes == 2 and 'MCI_SMC' in opt.category:
                      label_SMC.append(0)
                    else:
                      label_SMC.append(1)
                else:
                    continue
            else:
                continue
if csv_contain == True:
    fea_num = 3
np.random.seed(opt.manual_seed)
if len(data_health) > 0:
    data_health = np.array(data_health).reshape(int(HC_num/fea_num), fea_num)
    label_health = label_health[0:HC_num * fea_num:fea_num]
    health_list = np.concatenate((data_health, np.array(label_health).reshape(int(HC_num / fea_num), 1)), axis=1)
    np.random.shuffle(health_list)
    n_test_health = ceil(health_list.shape[0] * test_ratio)
    n_train_val_health = health_list.shape[0] - n_test_health
    train_val_list_health = health_list[0:n_train_val_health, :]
    test_list_health = health_list[n_train_val_health:health_list.shape[0], :]
if len(data_MCI) > 0:
    data_MCI = np.array(data_MCI).reshape(int(MCI_num/fea_num), fea_num)
    label_MCI = label_MCI[0:MCI_num * fea_num:fea_num]
    MCI_list = np.concatenate((data_MCI, np.array(label_MCI).reshape(int(MCI_num / fea_num), 1)), axis=1)
    np.random.shuffle(MCI_list)
    n_test_MCI = ceil(MCI_list.shape[0] * test_ratio)
    n_train_val_MCI = MCI_list.shape[0] - n_test_MCI
    train_val_list_MCI = MCI_list[0:n_train_val_MCI, :]
    test_list_MCI = MCI_list[n_train_val_MCI:MCI_list.shape[0], :]
if len(data_SMC) > 0:
    data_SMC = np.array(data_SMC).reshape(int(SMC_num/fea_num), fea_num)
    label_SMC = label_SMC[0:SMC_num * fea_num:fea_num]
    SMC_list = np.concatenate((data_SMC, np.array(label_SMC).reshape(int(SMC_num / fea_num), 1)), axis=1)
    np.random.shuffle(SMC_list)
    n_test_SMC = ceil(SMC_list.shape[0] * test_ratio)
    n_train_val_SMC = SMC_list.shape[0] - n_test_SMC
    train_val_list_SMC = SMC_list[0:n_train_val_SMC, :]
    test_list_SMC = SMC_list[n_train_val_SMC:SMC_list.shape[0], :]

kf = KFold(n_splits=opt.n_fold, shuffle=False)
n = 0
names = locals()
if len(data_health) > 0:
    for train_index, val_index in kf.split(train_val_list_health):
        n += 1
        names['train_fold%s_health'%n] = train_val_list_health[train_index]
        names['val_fold%s_health' % n] = train_val_list_health[val_index]
n = 0
if len(data_MCI) > 0:
    for train_index, val_index in kf.split(train_val_list_MCI):
        n += 1
        names['train_fold%s_MCI'%n] = train_val_list_MCI[train_index]
        names['val_fold%s_MCI' % n] = train_val_list_MCI[val_index]
n = 0
if len(data_SMC) > 0:
    for train_index, val_index in kf.split(train_val_list_SMC):
        n += 1
        names['train_fold%s_SMC'%n] = train_val_list_SMC[train_index]
        names['val_fold%s_SMC' % n] = train_val_list_SMC[val_index]
names2 = locals()
for i in range(1, n_fold+1):
    if len(data_health) > 0 and len(data_MCI) > 0 and len(data_SMC) > 0:
        names2['train_list_fold%s'%i] = np.vstack((names2.get('train_fold%s_health'%i), names2.get('train_fold%s_MCI'%i), names2.get('train_fold%s_SMC'%i)))
        names2['val_list_fold%s'%i] = np.vstack((names2.get('val_fold%s_health'%i), names2.get('val_fold%s_MCI'%i),  names2.get('val_fold%s_SMC'%i)))
        test_list = np.vstack((test_list_health, test_list_MCI, test_list_SMC))
    if len(data_health) > 0 and len(data_MCI) > 0 and len(data_SMC) == 0:
        names2['train_list_fold%s'%i] = np.vstack((names2.get('train_fold%s_health'%i), names2.get('train_fold%s_MCI'%i)))
        names2['val_list_fold%s'%i] = np.vstack((names2.get('val_fold%s_health'%i), names2.get('val_fold%s_MCI'%i)))
        test_list = np.vstack((test_list_health, test_list_MCI))
    if len(data_health) > 0 and len(data_MCI) == 0 and len(data_SMC) > 0:
        names2['train_list_fold%s'%i] = np.vstack((names2.get('train_fold%s_health'%i), names2.get('train_fold%s_SMC'%i)))
        names2['val_list_fold%s'%i] = np.vstack((names2.get('val_fold%s_health'%i), names2.get('val_fold%s_SMC'%i)))
        test_list = np.vstack((test_list_health, test_list_SMC))
    if len(data_health) == 0 and len(data_MCI) > 0 and len(data_SMC) > 0:
        names2['train_list_fold%s'%i] = np.vstack((names2.get('train_fold%s_MCI'%i), names2.get('train_fold%s_SMC'%i)))
        names2['val_list_fold%s'%i] = np.vstack((names2.get('val_fold%s_MCI'%i), names2.get('val_fold%s_SMC'%i)))
        test_list = np.vstack((test_list_MCI, test_list_SMC))
    np.random.seed(opt.manual_seed)
    np.random.shuffle(names2['train_list_fold%s'%i])
    np.random.shuffle(names2['val_list_fold%s'%i])

np.random.seed(opt.manual_seed)
np.random.shuffle(test_list)

csv_save_path = OsJoin(root, csv_save_dir)
if not os.path.exists(csv_save_path):
    os.makedirs(csv_save_path)

for i in range(1, n_fold+1):
    with open(OsJoin(csv_save_path, 'train_fold%s.csv'%i), 'w', newline='') as f:  
        f_csv = csv.writer(f)
        f_csv.writerows(names2.get('train_list_fold%s'%i))
    with open(OsJoin(csv_save_path, 'val_fold%s.csv'%i), 'w', newline='') as f:  
        f_csv = csv.writer(f)
        f_csv.writerows(names2.get('val_list_fold%s'%i))


with open(OsJoin(csv_save_path, 'test.csv'), 'w', newline='') as f: 
    f_csv = csv.writer(f)
    f_csv.writerows(test_list)
