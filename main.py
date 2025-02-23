# !pip install PyTDC
# !pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
# !pip install pytorch-lightning
# !pip install rdkit
# !pip install deepchem
import pytorch_lightning as pl
from torch_geometric.data import Data
import torch_geometric
import torch
import numpy as np
import pickle
from tdc import Evaluator
from torch_geometric.loader import LinkNeighborLoader, DataLoader
from sklearn.model_selection import train_test_split
from network_functions import *


with open('pathtodataset.pickle', 'rb') as f:
  new_dataset = pickle.load(f)

# An arbitrary collection of objects supported by pickle.

path = F"pathtotargets.pickle"
with open(path, 'rb') as f:
  # Pickle the 'data' dictionary using the highest protocol available.
  target_strat = pickle.load(f)


with open('pathtograph', 'rb') as f:
  drug_graph_unfeaturised_final= pickle.load(f)


with open('pathtodrugprahps.pickle', 'rb') as f:
  drug_graphs_finaltiprego= pickle.load(f)


from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(np.arange(len(new_dataset)),stratify=target_strat,test_size=0.2,random_state=37)



model_save_name = 'train_idx.pickle'
path = F"pathto/{model_save_name}"

import pickle


with open(path, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(train_idx, f, pickle.HIGHEST_PROTOCOL)


model_save_name = 'test_idx.pickle'
path = F"pathto/{model_save_name}"
import pickle


with open(path, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(test_idx, f, pickle.HIGHEST_PROTOCOL)




train_set = [new_dataset[i] for i in train_idx]
test_set = [new_dataset[i] for i in test_idx]


train_loader = DataLoader(train_set, batch_size=32,shuffle=True, drop_last=False)
test_loader = DataLoader(test_set,batch_size=32)





model, result = train_graph_classifier(
    model_name="GraphConv", c_hidden=128,  num_layers=2, dp_rate_linear=0.1,
)

model_save_name = 'classifier__30epochs_GCN.pt'
path = F"path_to_classifier/{model_save_name}"
torch.save(model, path)


oldmodel = torch.load(F"path_to_classifier/{model_save_name}.pt")
oldmodel.eval()

x=[[drug_graphs_finaltiprego[j], drug_graphs_finaltiprego[j], torch.tensor((j%86+1))]  for j in range(1706)]
loader = DataLoader(x, batch_size=32,shuffle=False, drop_last=False)
x_embedded=torch.empty(0,256)
for data in loader:
  x0=oldmodel(data)
  x_embedded=torch.cat((x_embedded,x0[0].detach().clone()),0)


drug_graph_GATfeature=drug_graph_unfeaturised_final
drug_graph_GATfeature.x=x_embedded


drug_graph_GATfeature_train=Data(x=drug_graph_GATfeature.x, edge_index=drug_graph_GATfeature.edge_index[:, train_idx], edge_attr=drug_graph_GATfeature.edge_attr[ train_idx])
drug_graph_GATfeature_test=Data(x=drug_graph_GATfeature.x, edge_index=drug_graph_GATfeature.edge_index[:, test_idx], edge_attr=drug_graph_GATfeature.edge_attr[ test_idx])



train_loader = LinkNeighborLoader(
    drug_graph_GATfeature_train,
    num_neighbors=[256],
    batch_size=512)



train_loader2=DataLoader([drug_graph_GATfeature_train], batch_size=32, shuffle=False)
test_loader = DataLoader([drug_graph_GATfeature_test], batch_size=32, shuffle=False)


model = LightningGCN()

device = 'gpu' if torch.cuda.is_available() else 'cpu'
num_epochs = 30


trainer = pl.Trainer(max_epochs=num_epochs,
                     accelerator=device)
trainer.fit(model, train_loader, val_dataloaders=train_loader2)

train_result = trainer.test(model, dataloaders=train_loader2, verbose=False)
test_result = trainer.test(model, dataloaders=test_loader, verbose=False)



model.eval()
targets_test=torch.empty(0)
with torch.no_grad():
        for data in train_loader2:
            output,_,_ = model(data)




model.eval()
targets_train=torch.empty(0)
with torch.no_grad():
        for data in test_loader:
            output_test,_,_ = model(data)
            preds=model.final_classifier(output_test,data.edge_index, 0 )
            loss = nn.CrossEntropyLoss()
            print( loss(preds, data.edge_attr-1))
            acc = (preds.argmax(dim=-1) == data.edge_attr-1).sum().float() / preds.shape[0]
            print(acc)
            F1 = Evaluator(name = 'Micro1')
            print(F1(preds.argmax(dim=-1), data.edge_attr-1))
            F2 = Evaluator(name = 'Macro1')
            print(F2(preds.argmax(dim=-1), data.edge_attr-1))
            kappa = Evaluator(name = 'Kappa')
            print(kappa(preds.argmax(dim=-1), data.edge_attr-1))
