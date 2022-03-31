import pickle
import torch
from pathlib import Path
import torchvision
from torch import nn
from datasets import SlideData
from collections import Counter
from tqdm import tqdm
import pandas as pd
import numpy as np
import sklearn
from PIL import Image
import os
from config import Config
from utils import get_dataset
config = Config()

save_all = False  
output_dst = 'clean_graph_updated_4allconn'
splits = ['train','test','val']
for split in splits:
    window_size = 224
    nonoverlap_factor = 2/3
    step_size = int(window_size * nonoverlap_factor)
    print(split)
    model_path = Path(config.val_model)
    parent_path = Path(config.validation_raw_src)
    ckpt = torch.load(f=str(model_path.joinpath('ckpt.pth')))
    model = torchvision.models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 3))
    model.load_state_dict(ckpt['model_state_dict'])
    model = nn.Sequential(*list(model.children())[:-1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    path_mean = [0.7725, 0.5715, 0.6842]
    path_std = [0.1523, 0.2136, 0.1721]
    transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=path_mean, std=path_std)])
    model.to(device)
    model.train(mode=False)
    train_dataset = SlideData(parent_path,transform,train=split,exclude=None,overall_info=config.val_raw_pkl) 
    dataloaders = {}
    dataloaders[split] = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=32,num_workers=8,shuffle=False)
    label_list = []
    pred_list = []
    score_list = []
    pos_list = []
    ids_list = []
    for idx, (inputs, labels, positions, wsi_ids) in enumerate(dataloaders[split]):
        train_inputs = inputs.to(device)
        train_labels = labels.to(device)
        with torch.set_grad_enabled(mode=False):
            train_outputs = model(train_inputs)
            train_preds = torch.argmax(train_outputs,dim=1)
            train_scores = [np.squeeze(torch.softmax(item,dim=0).detach().cpu().numpy()) for item in train_outputs]         
        pred_list.extend([item for item in train_preds.detach().cpu().numpy()])
        label_list.extend([item for item in train_labels.detach().cpu().numpy()])
        score_list.extend(train_scores)
        pos_info = [list(item.numpy()) for item in positions]
        for tissue_x,tissue_y,patch_x,patch_y in zip(pos_info[0],pos_info[1],pos_info[2],pos_info[3]):
            pos_list.append([(tissue_x//step_size + 1),(tissue_y//step_size + 1),patch_x,patch_y])
        ids_list.extend(list(wsi_ids.numpy()))

    id2label = pickle.load(open(config.val_raw_pkl,'rb'))[1]
    name_map = {'NotAnnotated':1,'Neoplastic':0,'Positive':2}
    graph_list = []
    for wsi_id in tqdm(list(set(ids_list))):
        wsi_label = name_map[id2label[wsi_id]]
        slide_pos = np.array(pos_list)[np.array(ids_list) == wsi_id]
        sx_min = np.min(np.array(pos_list)[np.array(ids_list) == wsi_id],axis=0)[0]
        sy_min = np.min(np.array(pos_list)[np.array(ids_list) == wsi_id],axis=0)[1]
        slide_pos[:,0] -= sx_min - 1
        slide_pos[:,1] -= sy_min - 1
        
        x = slide_pos[:,0] + slide_pos[:,2]
        y = slide_pos[:,1] + slide_pos[:,3]
        positive_score = np.array(score_list)[np.array(ids_list) == wsi_id]
        patch_label = np.array(label_list)[np.array(ids_list) == wsi_id]
        heat_img = np.zeros((max(x),max(y),512))
        label_img = np.ones((max(x),max(y)))
        for i in range(len(positive_score)):
            heat_img[x[i]-1,y[i]-1,:] = positive_score[i] * 255
            label_img[x[i]-1,y[i]-1] = patch_label[i]
        if (len(np.nonzero(heat_img)[0]) == len(np.nonzero(heat_img)[1])) & (len(np.nonzero(heat_img)[2]) == len(np.nonzero(heat_img)[1])):
            text = 'gg'
        else:
            print(text)
        if not os.path.exists(f'./{output_dst}/3cls_{split}'):
            os.makedirs(f'./{output_dst}/3cls_{split}')  
        if not os.path.exists(f'./{output_dst}/3cls_{split}/score/'):
            os.makedirs(f'./{output_dst}/3cls_{split}/score/')
        if not os.path.exists(f'./{output_dst}/3cls_{split}/label/'):
            os.makedirs(f'./{output_dst}/3cls_{split}/label/')      
        if save_all:        
            np.save(f'./{output_dst}/3cls_{split}/score/{wsi_id}_{wsi_label}',heat_img)
            np.save(f'./{output_dst}/3cls_{split}/label/{wsi_id}_{wsi_label}',label_img)
        else:
            graph_list.append(get_dataset(heat_img,label_img,wsi_id,wsi_label))
    if not save_all:
        with open(f'./{output_dst}/{split}_graphs.pkl','wb') as f:
            pickle.dump(graph_list,f)
