from pathlib import Path
import numpy as np
import os
import pickle

from torch import nn
from tqdm import tqdm
import torch
import torchvision

from config import Config
from datasets import SlideData
from utils import get_dataset


"""
This file is used to generate graph data from patches' features and positions.
The generated graphs will be used in main.py to train and evaluate the graph
"""


config = Config()
save_all = False  #Save the intermediate data or not
output_dst = config.output_dst  #destination to save graphs
splits = ['train', 'test', 'val']



for split in splits:
    #Input the parameters used to extract patch images#
    window_size = config.windows_size
    nonoverlap_factor = config.nonoverlap_factor
    step_size = int(window_size * nonoverlap_factor)
    ##################

    #Load the patch-level feature extractor and patch data
    model_path = Path(config.val_model)  #Patch-level feature extractor
    parent_path = Path(config.validation_raw_src)  #Path to small fixed-size patch images
    ckpt = torch.load(f=str(model_path.joinpath('ckpt.pth')))
    model = torchvision.models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, config.num_classes))
    model.load_state_dict(ckpt['model_state_dict'])
    model = nn.Sequential(*list(model.children())[:-1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path_mean = config.path_mean
    path_std = config.path_std
    transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=path_mean, std=path_std)])
    model.to(device)
    model.train(mode=False)
    train_dataset = SlideData(
        path = parent_path, transforms = transform,
        train=split,
        exclude=None,
        overall_info=config.val_raw_pkl) 
    dataloaders = {}
    dataloaders[split] = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)
    ##################

    #Extract patch-level features and match patch positions
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
            train_scores = [np.squeeze(
                torch.softmax(item, dim=0).detach().cpu().numpy()) for item in train_outputs]         
        pred_list.extend([item for item in train_preds.detach().cpu().numpy()])
        label_list.extend([item for item in train_labels.detach().cpu().numpy()])
        score_list.extend(train_scores)
        pos_info = [list(item.numpy()) for item in positions]
        for tissue_x, tissue_y, patch_x, patch_y in zip(*pos_info[:4]):
            pos_list.append([
                (tissue_x//step_size + 1),
                (tissue_y//step_size + 1),
                patch_x, patch_y])
        ids_list.extend(list(wsi_ids.numpy()))
    ##################

    #Construct graphs using patch features as node features 
    #and creating edges based on patch positions
    id2label = pickle.load(open(config.val_raw_pkl, 'rb'))[1]
    name_map = config.name_map

    pos_list_array = np.array(pos_list)
    ids_list_array = np.array(ids_list)
    graph_list = []
    for wsi_id in tqdm(list(set(ids_list))):
        wsi_label = name_map[id2label[wsi_id]]
        slide_pos = pos_list_array[ids_list_array == wsi_id]
        sx_min = np.min(pos_list_array[ids_list_array == wsi_id],axis=0)[0]
        sy_min = np.min(pos_list_array[ids_list_array == wsi_id],axis=0)[1]
        slide_pos[:,0] -= sx_min - 1
        slide_pos[:,1] -= sy_min - 1
        
        x = slide_pos[:,0] + slide_pos[:,2]
        y = slide_pos[:,1] + slide_pos[:,3]
        positive_score = np.array(score_list)[ids_list_array == wsi_id]
        patch_label = np.array(label_list)[ids_list_array == wsi_id]
        heat_img = np.zeros((max(x), max(y), 512))
        label_img = np.ones((max(x), max(y)))
        for i in range(len(positive_score)):
            heat_img[x[i]-1,y[i]-1,:] = positive_score[i] * 255
            label_img[x[i]-1,y[i]-1] = patch_label[i]

        if not os.path.exists(f'./{output_dst}/3cls_{split}'):
            os.makedirs(f'./{output_dst}/3cls_{split}')  
        if not os.path.exists(f'./{output_dst}/3cls_{split}/score/'):
            os.makedirs(f'./{output_dst}/3cls_{split}/score/')
        if not os.path.exists(f'./{output_dst}/3cls_{split}/label/'):
            os.makedirs(f'./{output_dst}/3cls_{split}/label/')      
        if save_all:
            specific_output_dir = Path('.').joinpath(output_dst).joinpath(f'3cls_{split}')
            np.save(specific_output_dir.joinpath('score').joinpath(f'{wsi_id}_{wsi_label}', heat_img))
            np.save(specific_output_dir.joinpath('label').joinpath(f'{wsi_id}_{wsi_label}', label_img))        
        else:
            graph_list.append(get_dataset(heat_img, label_img, wsi_id, wsi_label))
    if not save_all:
        with open(f'./{output_dst}/{split}_graphs.pkl','wb') as f:
            pickle.dump(graph_list, f)
    ##################
