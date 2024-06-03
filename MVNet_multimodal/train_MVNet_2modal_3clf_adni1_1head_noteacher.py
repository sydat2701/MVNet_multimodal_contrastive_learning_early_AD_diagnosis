# from teacher.teacher import Teacher

import os
import warnings
warnings.filterwarnings("ignore")
import torch
from torch._C import set_flush_denormal
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from teacher.data_loader_is import Feeder
from torch.utils.data import Dataset, DataLoader
from configs import get_3DReg_config
import gc
import copy

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    lr = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)*0.8
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("Current lr: ", lr)


import time
from early_stopper import EarlyStopper
from eval_metrics import report


def get_task_data(X, y, task):
    neg_idx = pos_idx = -1
    X_return = []
    y_return = []
    if task == 'cn_ad':
        neg_idx = 0
        pos_idx = 3
    elif task == 'cn_mci':
        neg_idx = 0
        pos_idx = (1, 2)


    for i, class_id in enumerate(y):
        if class_id == neg_idx:
            X_return.append(X[i])
            y_return.append(0)
        
        if isinstance(pos_idx, tuple):
            if class_id in pos_idx:
                X_return.append(X[i])
                y_return.append(1)
                
        elif class_id == pos_idx:
            X_return.append(X[i])
            y_return.append(1)
            
    return np.asarray(X_return), np.asarray(y_return)

def get_subs_labels(subs, y, task):
    neg_idx = pos_idx = -1
    sub_return = []
    y_return = []
    if task == 'cn_ad':
        neg_idx = 0
        pos_idx = 3
    elif task == 'cn_mci':
        neg_idx = 0
        pos_idx = (1, 2)


    for i, class_id in enumerate(y):
        if class_id == neg_idx:
            sub_return.append(subs[i])
            y_return.append(0)
        
        if isinstance(pos_idx, tuple):
            if class_id in pos_idx:
                sub_return.append(subs[i])
                y_return.append(1)
                
        elif class_id == pos_idx:
            sub_return.append(subs[i])
            y_return.append(1)
            
    return np.asarray(sub_return), np.asarray(y_return)


kf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=2022)
latest_saved_weight =None

contras_weight_path = '/home/id202388544/dat/MVNet_fol/MVNet_surfteacher_contras/contras_base_extract_net/weights_contras_base_adni1'

def train(num_epoch, device, criterion, config, weight_path, lr, dataloader_train, dataloader_val, \
          fol_idx, log_path, is_cosinelr, is_early_stop, batch_size):

    fuse_loss = nn.MSELoss()

    
    from extract_net_extend_2modal_adni1_1head_noteacher import MVNetContras
    model = MVNetContras(contras_weight_path, str(fol_idx))
    model = nn.DataParallel(model)
    model.to(device)

    lr = lr

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)

    if is_cosinelr:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 20, eta_min = 10e-5)

    if is_early_stop:
        early_stopper = EarlyStopper(patience=15, min_delta=0)
    #best_model_weights = copy.deepcopy(model.state_dict())
    best_val_bacc = -99999999
    num100 = 0
    for epoch in range(num_epoch):
        print("Epoch: {} **************".format(epoch+1))
        epoch_train_acc = 0
        epoch_train_loss=0
        epoch_fuse_lossx = 0
        epoch_fuse_v_lossx = 0
        epoch_fuse_lossy = 0
        epoch_fuse_v_lossy = 0

        num_iters_train=0
        model.train()

        if not is_cosinelr:
            if epoch != num_epoch//2:
                adjust_learning_rate(optimizer, epoch, num_epoch, lr, flag = False)
            else:
                adjust_learning_rate(optimizer, epoch, num_epoch, lr, flag = True)
        

        for inputs, labels in dataloader_train:

            mri = inputs[1][0].to(device).float()
            fdg = inputs[1][1].to(device).float()
            x = inputs[0].to(device).float()
            labels = labels.to(device).float()
            num_iters_train += 1
            
            y_pred, y_pred_mri, y_pred_fdg = model(mri, fdg)
            

            loss= criterion(y_pred, labels) + criterion(y_pred_mri, labels) + criterion(y_pred_fdg, labels)


            total_loss = loss 

            epoch_train_loss += loss.item()

            total_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            y_pred = (y_pred+y_pred_mri+y_pred_fdg)/3

            y_pred_ = torch.round(y_pred)
            equality = (labels.data == y_pred_.data)
            train_acc = equality.type_as(torch.FloatTensor()).mean()
            #-------------------------------------------------------------------
            epoch_train_acc +=train_acc.item()

        train_accuracy = epoch_train_acc / (num_iters_train)
        train_loss = epoch_train_loss / num_iters_train
        if np.round(train_accuracy, 2) ==1.0:
            num100 +=1
        
        
        if is_cosinelr:
            sched.step()
            for param_group in optimizer.param_groups:
                print("Current lr: ", param_group['lr'])
        
        log_train = open(os.path.join(log_path, 'train.txt'), 'a')
        log_train.write('Epoch {}: train_acc {}, train_loss {}\n'.format(epoch, train_accuracy, \
                                                                       (train_loss/num_iters_train)))
        log_train.close()

        epoch_train_loss =0
        epoch_train_acc=0
        #print("epoch", (epoch+1))
        print("train_acc:", np.round(train_accuracy, 3), "fuse_lossx:", np.round(epoch_fuse_lossx/num_iters_train, 3), \
              "fuse_v_lossx:", np.round(epoch_fuse_v_lossx/num_iters_train, 3), "fuse_lossy:", np.round(epoch_fuse_lossy/num_iters_train, 3), "fuse_v_lossy:", np.round(epoch_fuse_v_lossy/num_iters_train, 3), "train_loss:", np.round(train_loss,3))
        num_iters_train=0
        #---------------------------------------VALIDATION---------------------------
        epoch_val_acc = 0
        epoch_val_loss=0
        num_iters_val = 0

        list_labels = []
        list_preds = []

        model.eval()

        for inputs, labels in dataloader_val:
            mri = inputs[1][0].to(device).float()
            fdg = inputs[1][1].to(device).float()
            x = inputs[0].to(device).float()
            labels = labels.to(device).float()

            num_iters_val += 1
            with torch.no_grad():
                y_pred, y_pred_mri, y_pred_fdg = model(mri, fdg)
                
                loss = criterion(y_pred, labels) + criterion(y_pred_mri, labels) + criterion(y_pred_fdg, labels)

            epoch_val_loss += loss.item()

            y_pred = (y_pred+y_pred_mri+y_pred_fdg)/3
            y_pred_ = torch.round(y_pred)
            equality = (labels.data == y_pred_.data)
            val_acc = equality.type_as(torch.FloatTensor()).mean()

            
            
            list_labels.extend(labels.cpu().detach().numpy().reshape(-1).tolist())
            list_preds.extend(y_pred.cpu().detach().numpy().reshape(-1).tolist())

            epoch_val_acc +=val_acc.item()
        val_accuracy = epoch_val_acc / (num_iters_val)
        val_loss = epoch_val_loss / num_iters_val

        report_dict =  report(list_labels, list_preds)
        
        epoch_val_loss =0
        epoch_val_acc=0
        num_iters_val=0

        print("val_acc:", val_accuracy, "val_loss:", val_loss)
        print("Report in detail: ", report_dict)


        log_val = open(os.path.join(log_path, 'val.txt'), 'a')
        log_val.write('Epoch {}: val_acc {}, val_loss {}\n'.format(epoch, val_accuracy, \
                                                                       val_loss))
        log_val.close()

        sen, spe, bacc, auc = report_dict['tpr'], report_dict['tnr'], report_dict['bacc'], report_dict['auc']
        if bacc> best_val_bacc:
            best_val_bacc = bacc
            global latest_saved_weight
            if latest_saved_weight !=None:
                os.remove(latest_saved_weight)
            tmp_path = weight_path+'/'+'{:.4f}acc_{:.4f}bacc_{:.4f}auc_{:.4f}sen_{:.4f}spe'.format( \
                val_accuracy, bacc, auc, sen, spe)+ '_fol'+str(fol_idx)+'.pth'
            latest_saved_weight = tmp_path
            torch.save(model.state_dict(), tmp_path)
            print("----------------------->saved model<--------------------")
        
        if num100 >=25:
            break
        #sched.step()"""

import glob


import random
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
import shutil
def main():
    #ori_seed: 53
    seed = 65
    seed_everything(seed) #seed 68
    log_path = f'./logs/ADNI1_fdg_mri_3clf_contras_1head_noteacher_seed{seed}'
    num_folds = 5
    num_epoch=120
    is_cosinelr = True
    is_early_stop = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.BCELoss()

    config = get_3DReg_config()
    weight_path = f'teacher/weights/ADNI1_fdg_mri_3clf_contras_1head_noteacher_seed{seed}'

    data_path = '/home/id202388544/dat/Data_ADNI1_ANTS'
    data_split_path = 'adni1_labels'
    batch_size = 4
    lr = 0.0003
    tasks = ['cn_mci']

    X, y = np.load('/home/id202388544/dat/surface/data/ADNI1/data_fsaverage_nopvc_ico2.npy').astype('float32'), np.load('/home/id202388544/dat/surface/data/ADNI1/labels.npy')
    subs = np.load('./teacher/surf_ids/sub.npy', allow_pickle=True)
    ids = np.load('./teacher/surf_ids/labels.npy', allow_pickle=True)


    
    
    for i, task in enumerate(tasks):
        print("--------------------------------------------{}-----------------------------------------------------".format(task))
        X_ , y_, = get_task_data(X, y, task)
        #print("Training data: ", list_subs.shape, list_y.shape)
        print(f"Training data: {X_.shape} {y_.shape}" )
        kf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=2022)
        fold = 0
        BACC = []
        SEN  = []
        SPE  = []
        AUC_SCORE  = []
        fol_idx = 0

        list_subs, list_y = get_subs_labels(subs, ids, task)

        for train_idx, test_idx in kf.split(X_, y_):
            subs_train = list_subs[train_idx]
            subs_test = list_subs[test_idx]
            global latest_saved_weight
            latest_saved_weight = None
            
            gc.collect()
            fold += 1
            fold_weight_path = os.path.join(weight_path, task)
            fold_log_path = os.path.join(log_path, task)

            os.makedirs(fold_log_path, exist_ok=True)
            os.makedirs(fold_weight_path, exist_ok=True)

            print(f"*************************************FOLD: {fol_idx}**************************************************")

            X_train, y_train = X_[train_idx], np.expand_dims(y_[train_idx], axis=-1)
            X_test, y_test = X_[test_idx], np.expand_dims(y_[test_idx], axis=-1)
            train_feeder = Feeder(X_train, y_train, data_path, subs_train) #data shape: (B, C, P, V) for data Q
            val_feeder = Feeder(X_test, y_test, data_path, subs_test)

            dataloader_train = DataLoader(dataset=train_feeder, batch_size=batch_size, shuffle= True)
            dataloader_val = DataLoader(dataset=val_feeder, batch_size=batch_size, shuffle= False)

            train(num_epoch, device, criterion, config, weight_path, lr, dataloader_train, dataloader_val, \
                    fol_idx, log_path, is_cosinelr, is_early_stop, batch_size)
            fol_idx +=1

    

if __name__=='__main__':
    main()


