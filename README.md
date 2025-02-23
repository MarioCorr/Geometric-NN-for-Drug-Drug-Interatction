# DDI problem
Definition: Drug-drug interactions occur when two or more drugs interact with each other. These could result in a range of outcomes from reducing the efficacy of one or both drugs to dangerous side effects such as increased blood pressure or drowsiness. Polypharmacy side-effects are associated with drug pairs (or higher-order drug combinations) and cannot be attributed to either individual drug in the pair. This task is to predict the interaction between two drugs.

Dataset Description: DrugBank drug-drug interaction dataset is manually sourced from FDA/Health Canada drug labels as well as primary literature. It has 86 interaction types.

More information on the dataset [here](https://tdcommons.ai/multi_pred_tasks/ddi)

Given the SMILES strings of two drugs, predict their interaction type. This project uses a two steps graph neural network approach:
<ul>
 <li>first from the graph representation of the drugs, we use two layers Graph Convolutional networks on the two drugs, sharing the same weights
 then classification is performed using a fully connected layer.</li>
 <li>we use the the GCN layers that were trained in the previous step to get a representation of the drugs. Then using
 the The [relational graph convolutional operator](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.RGCNConv.html)
 from the "Modeling Relational Data with Graph Convolutional Networks" paper is used then to classify the edges.</li>
 </ul>

## Notes
The code is a distillation of the code of a jupyter notebook. The notebook used to call back and forth the saved files in a google drive folder, in order to keep track of the two step approach.
For this reason some of the variables might be ranamed more than once.
