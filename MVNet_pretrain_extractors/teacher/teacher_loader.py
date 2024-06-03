import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys
import random
import nibabel as nib
import cv2
import os
import pandas as pd 
sys.path.extend(['../'])
import glob

class Feeder(Dataset):
    def __init__(self, data_path, cn_subs, mci_subs, ad_subs, modal):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.modal = modal
        self.data_path = data_path
        self.all_subs = np.concatenate([cn_subs, mci_subs, ad_subs])
        self.labels = np.concatenate([np.zeros((cn_subs.shape[0])), np.ones((mci_subs.shape[0])), np.ones((ad_subs.shape[0]))*2])
        
        #permute all
        rd_arr = np.random.randint(0, len(self.labels), size = len(self.labels))
        np.random.shuffle(rd_arr)
        self.all_subs = self.all_subs[rd_arr]
        self.labels = self.labels[rd_arr]


    
    def normalize_data(self, data):
        max_val, min_val = np.amax(data), np.amin(data)

        data = (data-min_val)/(max_val-min_val)
        return data

        
    def to_categorical(self, num):
        arr = np.zeros((self.num_cls))
        arr[num-1] = 1
        return torch.from_numpy(arr)
        

    def load_data(self, sub, label):
        #path =os.path.join(self.data_path, sub)
        path = os.path.join(self.data_path, sub, sub+'_{}_mask_norm_crop.nii.gz'.format(self.modal)) 
    
        img = nib.load(path).get_fdata()

        
        # mri_img = self.normalize_data(mri_img)
        #x = [torch.from_numpy(pet_img).unsqueeze(0)]
        x = [torch.from_numpy(img).unsqueeze(0)]
        
            
        y = torch.from_numpy(np.array([label]))
        #y = torch.from_numpy(np.array(self.to_categorical(self.label_to_numeric(label))))

        return x, y

    def __len__(self):
        return len(self.labels)


    def label_to_numeric(self, label):
        if label=='CN':
            return 0
        if label=='Dementia':
            return 1


    def __getitem__(self, index):
        # curr_mri = self.list_all_mri[index]
        curr_label = self.labels[index]
        curr_sub = self.all_subs[index]
        # curr_pet = self.list_all_pet[index]

        x, y = self.load_data(curr_sub, curr_label)
        return x, y



def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

