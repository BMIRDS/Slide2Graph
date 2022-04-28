from matplotlib import pyplot as plt
from pathlib import Path
import imp
import math
import numpy as np
import pickle

from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support as score
from torch_geometric import utils as gutils
from torch_geometric.data import DataLoader as gDataLoader
import torch
import torch_geometric.data as gdata
import torch_geometric.utils as gutils

#from utils import get_dataset
from config import Config
from models import GraphCls


config = Config()

train_graphs = pickle.load(open(config.train_graphs, 'rb'))
test_graphs = pickle.load(open(config.test_graphs, 'rb'))
val_graphs = pickle.load(open(config.val_graphs, 'rb'))

mode = config.mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loader = gDataLoader(train_graphs, batch_size=config.batch_size, shuffle=True)
vloader = gDataLoader(val_graphs, batch_size=len(val_graphs), shuffle=False)
tloader = gDataLoader(test_graphs, batch_size=len(test_graphs), shuffle=False)
model = GraphCls(hidden_channels=config.hidden_size)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay) 
criterion = torch.nn.CrossEntropyLoss()

scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    'min',
    factor=0.8,
    patience=5,
    threshold=0.0001)

def run():
    ##Train the model
    model.to(device)
    criterion.to(device)
    train_loss = []
    val_acc = []
    lr_list = []
    if mode == 'train':
        for epoch in range(1, config.epochs+1):
            train_running_loss = 0.0
            for gb in loader:
                model = model.train()
                optimizer.zero_grad()
                out = model(gb.x.to(device), gb.edge_index.to(device), gb.edge_attr.to(device), gb.batch.to(device))
                loss = criterion(out, gb.graph_y.to(device)) 
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_running_loss += loss.item() * gb.graph_y.size(0)
            current_lr = optimizer.param_groups[0]['lr']
            lr_list.append(current_lr)
            print(f'training loss: {train_running_loss/len(train_graphs)}')   
            train_loss.append(train_running_loss/len(train_graphs)) 
            for gb in vloader:
                with torch.no_grad():
                    model = model.eval()
                    out = model(gb.x.to(device), gb.edge_index.to(device), gb.edge_attr.to(device), gb.batch.to(device))
                pred = np.array(torch.argmax(out, dim=1).cpu())
                true = np.array(gb.graph_y)
                valid_acc = accuracy_score(true, pred)
                print(f'{epoch}: {current_lr} valid_accuracy:{valid_acc}', end='; ')
                val_acc.append(valid_acc)
            scheduler1.step(loss.item())

        #Evaluate the model
        print('Evaluation:')
        loader = gDataLoader(train_graphs, batch_size=len(train_graphs), shuffle=False)
        for gb in loader:
            model = model.eval()
            out = model(gb.x.to(device), gb.edge_index.to(device), gb.edge_attr.to(device), gb.batch.to(device))
            pred = np.array(torch.argmax(out, dim=1).cpu())
            true = np.array(gb.graph_y)
        score_list = softmax(out.cpu().detach().numpy(),axis=1)
        print(confusion_matrix(true, pred))
        print(score(true, pred))
        loader = gDataLoader(val_graphs, batch_size=len(val_graphs), shuffle=False)
        for gb in loader:
            print(gb.slide_index)
            model = model.eval()
            out = model(gb.x.to(device), gb.edge_index.to(device), gb.edge_attr.to(device), gb.batch.to(device))
            pred = np.array(torch.argmax(out,dim=1).cpu())
            true = np.array(gb.graph_y)
        print(confusion_matrix(true, pred))
        print(score(true,pred))
        torch.save(model.state_dict(), config.model_path)

    else:
        loader = gDataLoader(test_graphs, batch_size=len(test_graphs), shuffle=False)
        for gb in loader:
            model.load_state_dict(torch.load(config.model_path))
            model = model.eval()
            out = model(gb.x.to(device), gb.edge_index.to(device), gb.edge_attr.to(device), gb.batch.to(device))
            pred = np.array(torch.argmax(out, dim=1).cpu())
            true = np.array(gb.graph_y)
        print(confusion_matrix(true, pred))
        print(score(true, pred))


if __name__ == '__main__':
    run()
