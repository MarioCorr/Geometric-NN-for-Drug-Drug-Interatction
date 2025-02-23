!pip install PyTDC
!pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
!pip install pytorch-lightning
import pytorch_lightning as pl
!pip install rdkit
!pip install deepchem
import deepchem as dc
from tdc.multi_pred import DDI
import torch_geometric
import torch
from tdc.chem_utils import MolConvert
from tqdm.notebook import tqdm

from matplotlib import pyplot as plt
dB = DDI(name = 'DrugBank')
from tdc.utils import get_label_map
a=(get_label_map(name = 'DrugBank', task = 'DDI'))
g=a.items()



my_file = open("pathtosavedsmilesofmolecules", "r")

smiles_sporchi= my_file.read()
smiles = smiles_sporchi.split("\n")
for j in range(len(smiles)):
  smiles[j]=smiles[j].split('\t')






converter = MolConvert(src = 'SMILES', dst = 'ECFP6')
drugs_graphs_finaltiprego=[]
failed_molecules=[]

for j in range(len(smiles)):
  data=smiles[j]
  try:

    features=converter(data[1])

    drugs_graphs_finaltiprego.append(torch.unsqueeze(torch.from_numpy(features),0))


  except:
    failed_molecules.append(data)


for j in tqdm(range(len(drugs_graphs_finaltiprego))):
    if len(drugs_graphs_finaltiprego[j].edge_index.size()) == 1:
        drugs_graphs_finaltiprego[j].edge_index = torch.empty([2, 0], dtype=torch.long)