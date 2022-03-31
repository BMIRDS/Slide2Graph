import torch
from torch.utils.data import Dataset
import pickle
from pathlib import Path
import os
import PIL
from PIL import Image
import numpy as np
from config import Config
config = Config()


def get_slide_path(train_path, parent_path):
    '''
    args:
    train_path: a list of WSI names
    parent_path: stem path
    '''
    patch_path = []
    patch_label = []
    patch_position = []
    parent_slide = []
    for item in train_path:
        slide_folder = parent_path.joinpath(item)
        for class_name in os.listdir(slide_folder):
            class_folder = slide_folder.joinpath(class_name)
            for patch in os.listdir(class_folder):
                patch_path.append(class_folder.joinpath(patch))
                patch_label.append(class_name)
                patch_position.append([int(i) for i in patch[:-4].split('_')])
                parent_slide.append(item)
    return patch_path,patch_label,patch_position,parent_slide


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class SlideData(Dataset):
    def __init__(self,path,transforms=None,train='train',exclude=None,overall_info=config.dst_pkl):
        self.transforms = transforms
        self.path = path
        overall_info = pickle.load(open(overall_info,'rb'))
        if exclude is not None:
            overall_info[1] = {k:v for k,v in overall_info[1].items() if v!=exclude}
            overall_info[0] = {k:v for k,v in overall_info[0].items() if k in overall_info[1].keys()}
            overall_info[2] = {k:v for k,v in overall_info[2].items() if k in overall_info[1].keys()}
        self.overall_info = overall_info #[id2slide,id2label,id2split]
        self.slide2id = {v: k for k, v in overall_info[0].items()}
        self.classes, self.class_to_idx = self._find_classes(overall_info[1])
        wsi_id = overall_info[0].keys()
        train_id = [item for item in wsi_id if overall_info[2][item]=='train']
        val_id = [item for item in wsi_id if overall_info[2][item]=='val']
        train_plus_val_id = train_id + val_id
        test_id = [item for item in wsi_id if overall_info[2][item]=='test']
        if train == 'train':
            # path, label, position, parent slide
            self.plpp = get_slide_path([overall_info[0][item] for item in train_id],path) 
            self.current_id = train_id
        elif train == 'val':
            self.plpp = get_slide_path([overall_info[0][item] for item in val_id],path)
            self.current_id = val_id
        elif train == 'trainplusval':
            self.plpp = get_slide_path([overall_info[0][item] for item in train_plus_val_id],path)
            self.current_id = train_plus_val_id
        else:
            self.plpp = get_slide_path([overall_info[0][item] for item in test_id],path)
            self.current_id = test_id
    def _find_classes(self,id2label):
        classes = list(set(id2label.values()))
        classes.sort()
        class_to_idx = {cls_name:i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self,index):
        sample = pil_loader(self.plpp[0][index])
        if self.transforms is not None:
            sample = self.transforms(sample)
        target = self.class_to_idx[self.plpp[1][index]] 
        return sample, target, self.plpp[2][index], self.slide2id[self.plpp[3][index]]
    def __len__(self):
        return len(self.plpp[0])
    def pick_WSI(self,wsi_id):
        self.plpp = get_slide_path([self.overall_info[0][item] for item in [wsi_id]],self.path)
        self.current_id = wsi_id







