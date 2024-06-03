# from teacher.teacher import Teacher

import os

import torch
from torch._C import set_flush_denormal
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from teacher.teacher_loader import Feeder
from torch.utils.data import Dataset, DataLoader

import copy
from cosine_lr.cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from tqdm import tqdm
from losses import SupConLoss

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    lr = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)*0.8
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("Current lr: ", lr)


import time
from early_stopper import EarlyStopper
from eval_metrics import report

def get_subs_labels_for_fol(subs, y, fol):
    val_subs = np.load('teacher/val_sub/split_sub_val_ADNI1/'+str(fol)+'.npy', allow_pickle=True)
   
    cn_subs = []
    mci_subs = []
    ad_subs = []


    for i, class_id in enumerate(y):
        #do not take the sub existed in validation set
        if subs[i] in val_subs:
            continue

        if class_id == 0:
            cn_subs.append(subs[i])
        if class_id == 1 or class_id ==2:
            mci_subs.append(subs[i])
        if class_id==3:
            ad_subs.append(subs[i])
            
    return np.asarray(cn_subs), np.asarray(mci_subs), np.asarray(ad_subs)

def cosine_scheduler(optimizer, initial_lr, min_lr, epochs_per_cycle, epoch):
    cycle = 50
    
    lr = min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * (epoch % cycle) / cycle)) / 2
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("Current lr: ", lr)

global_path = None
def train(num_epoch, device, criterion, weight_path, lr, dataloader_train, \
          fol_idx, log_path, is_cosinelr, is_early_stop, batch_size):
    from extract_net_extend import MVNet
    model = MVNet()
    model = nn.DataParallel(model)
    model.to(device)

    lr = lr

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)

    if is_cosinelr:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 20, eta_min = 10e-5)

    if is_early_stop:
        early_stopper = EarlyStopper(patience=15, min_delta=0)
    #best_model_weights = copy.deepcopy(model.state_dict())
    best_train_loss= 99999999
    num100 = 0
    for epoch in range(num_epoch):
        print("Epoch: {} **************".format(epoch+1))
        epoch_train_acc = 0
        epoch_train_loss=0
        num_iters_train=0
        model.train()

        pbar_train = tqdm(dataloader_train)
        for inputs, labels in pbar_train:

            x = inputs[0].to(device).float()

            labels = labels.to(device).float()
            num_iters_train += 1
            
            features = model(x)
            features = features.unsqueeze(1)
            

            loss= criterion(features, labels)
            epoch_train_loss += loss.item()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            pbar_train.set_description(f"Loss: {epoch_train_loss/num_iters_train:.4f}")
        train_loss = epoch_train_loss / num_iters_train
        
        if is_cosinelr:
            sched.step()
            for param_group in optimizer.param_groups:
                print("Current lr: ", param_group['lr'])
        
        log_train = open(os.path.join(log_path, 'train.txt'), 'a')
        log_train.write('Epoch {}: train_loss {}\n'.format(epoch, \
                                                        (train_loss/num_iters_train) ))
        log_train.close()

        epoch_train_loss =0
        epoch_train_acc=0
        num_iters_train=0
        print("train_loss:", train_loss)

        global global_path
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            if global_path !=None:
                os.remove(global_path)
            global_path = os.path.join(weight_path, 'weight_epoch{}_loss{}.pth'.format(str(epoch), str(np.round(train_loss, 5))))
            torch.save(model.state_dict(), global_path)
            print("----------------------->saved model<--------------------")

        if epoch >= num_epoch-1:
            torch.save(model.state_dict(), os.path.join(weight_path, 'weight_epoch{}_loss{}.pth'.format(str(epoch), str(np.round(train_loss, 5)))))
            print("----------------------->saved model<--------------------")

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

from sklearn.model_selection import StratifiedKFold


def main():
    #ori_seed: 53
    seed = 51
    seed_everything(seed)
    log_path = './logs/ADNI1/seed{}_att'.format(seed)
    num_folds = 5
    num_epoch=150
    is_cosinelr = False
    is_early_stop = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = SupConLoss(temperature=0.1)

    weight_path = 'teacher/weights/ADNI1/seed{}_att'.format(seed)
    if os.path.exists('./log_val_pred.txt'):
        os.remove('./log_val_pred.txt')


    data_path = '/home/id202388544/dat/Data_ADNI1_ANTS'

    batch_size = 16
    lr = 0.001
    task = 'cn_mci'

    X, y = np.load('/home/id202388544/dat/surface/data/ADNI1/data_fsaverage_nopvc_ico2.npy').astype('float32'), np.load('/home/id202388544/dat/surface/data/ADNI1/labels.npy')
    subs = np.load('./teacher/surf_ids/sub.npy', allow_pickle=True)
    ids = np.load('./teacher/surf_ids/labels.npy', allow_pickle=True)


    list_fols = [0,1,2,3,4]
    fol_idx = 0
    list_modals = ['fdg', 'MRI']

    for modal in list_modals:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>Modal {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(modal))
        cn_subs, mci_subs, ad_subs = get_subs_labels_for_fol(subs, y, fol_idx)
        
        for fol_idx in list_fols:
            print("--------------------------------------------FOLD {}-----------------------------------------------".format(fol_idx))
            global latest_saved_weight
            latest_saved_weight = None

            fol_log_path = os.path.join(log_path, str(fol_idx), modal)
            fol_weight_path = os.path.join(weight_path, str(fol_idx), modal)
            os.makedirs(fol_log_path, exist_ok=True)
            os.makedirs(fol_weight_path, exist_ok=True)


            train_feeder = Feeder(data_path, cn_subs, mci_subs, ad_subs, modal)

            dataloader_train = DataLoader(dataset=train_feeder, batch_size=batch_size, shuffle= True)
            print("Done loading data")
            train(num_epoch, device, criterion, fol_weight_path, lr, dataloader_train, \
                fol_idx, fol_log_path, is_cosinelr, is_early_stop, batch_size)
        

    

if __name__=='__main__':
    main()

