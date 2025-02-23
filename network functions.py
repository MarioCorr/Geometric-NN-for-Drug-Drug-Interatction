!pip install PyTDC
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
!pip install pytorch-lightning
import pytorch_lightning as pl
import torch_geometric
import torch
from torch_geometric.nn import GCN, RGCNConv, GAT, BatchNorm
import torch.nn as nn
from torch_geometric.nn import GCN
import torch.nn.functional as F





class GraphGNNModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out=256, dp_rate_linear=0.2, **kwargs):
        """
        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of output features (usually number of classes)
            dp_rate_linear: Dropout rate before the linear layer (usually much higher than inside the GNN)
            kwargs: Additional arguments for the GNNModel object
        """
        super().__init__()
        self.GNN1 = GCN(in_channels=c_in, hidden_channels=c_hidden, out_channels=c_hidden, **kwargs)
        self.GNN2 = GCN(in_channels=c_hidden, hidden_channels=c_hidden, out_channels=c_hidden, **kwargs)
        self.head1 = nn.Sequential(nn.Dropout(dp_rate_linear), nn.Linear(c_hidden, c_out), nn.ReLU())
        self.head2 = nn.Sequential(nn.Dropout(dp_rate_linear), nn.Linear(c_out, c_out), nn.ReLU())
    def forward(self, x, edge_index, batch_idx):
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            batch_idx: Index of batch element for each node
        """

        x = F.relu(self.GNN1(x, edge_index))
        x = F.relu(self.GNN2(x, edge_index))
        x = torch_geometric.nn.global_max_pool(x, batch_idx)
        x = self.head1(x)
        x = self.head2(x)

        return x





class GraphLevelGNN(pl.LightningModule):
    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = GraphGNNModel(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 86))

    def forward(self, data, mode="train"):
        x0, edge_index0, batch_idx0 = data[0].x, data[0].edge_index, data[0].batch
        x1, edge_index1, batch_idx1 = data[1].x, data[1].edge_index, data[1].batch
        x0 = self.model(x0, edge_index0, batch_idx0)

        x0 = x0.squeeze(dim=-1)
        x1 = self.model(x1, edge_index1, batch_idx1)
        x1 = x1.squeeze(dim=-1)
        x = torch.cat((x0, x1), 1)
        x= self.head(x)
        preds = x.argmax(dim=-1)
        data[2] = data[2].squeeze(dim=-1)-1
        loss = self.loss_module(x, data[2])
        acc = (preds == data[2]).sum().float() / preds.shape[0]

        return x0, preds, loss, acc

    def configure_optimizers(self):
        # High lr because of small dataset and small model
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.0)
        return optimizer

    def training_step(self, batch, batch_idx):
        _,_, loss, acc= self.forward(batch, mode="train")


        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, loss, acc = self.forward(batch, mode="val")
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def test_step(self, batch, batch_idx):
        x0, preds, loss, acc = self.forward(batch, mode="test")
        self.log("test_loss", acc)
        self.log("test_acc", acc)
        return





from pytorch_lightning.callbacks import ModelCheckpoint
def train_graph_classifier(model_name, **model_kwargs):
    pl.seed_everything(43)
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=30,
    )
    trainer.logger._default_hp_metric = None
    pl.seed_everything(37)
    model = GraphLevelGNN(
           c_in=new_dataset[0][0].num_node_features,
            **model_kwargs,
        )
    trainer.fit(model, train_loader, val_dataloaders=test_loader)
    train_result = trainer.test(model, dataloaders=train_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"train": train_result, "test": test_result}
    return model, result



class GraphRGCNodel(nn.Module):
    def __init__(self, c_in, c_hidden=64, c_out=16, dp_rate_linear=0.1, **kwargs):
        """
        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of output features (usually number of classes)
            dp_rate_linear: Dropout rate before the linear layer (usually much higher than inside the GNN)
            kwargs: Additional arguments for the GNNModel object
        """
        super().__init__()
        self.GNN1 = RGCNConv(in_channels=c_in,  out_channels=64, num_relations=86)
        self.GNN2 = RGCNConv(in_channels=c_in,  out_channels=256,num_relations=86)
        self.GNN3 = RGCNConv(in_channels=64, out_channels=32,  num_relations=86)
        self.GNN4 = RGCNConv(in_channels=32, out_channels=32,  num_relations=86)
        self.drop1  = nn.Dropout(p=0.3)
        self.drop2  = nn.Dropout(p=0.4)
        self.drop3  = nn.Dropout(p=0.3)
        self.drop4  = nn.Dropout(p=0.3)
        self.GNN31 = GAT(in_channels=c_in, hidden_channels=c_hidden, out_channels=64, num_layers=1, edge_dim=86, heads=3)
        self.GNN32 = GAT(in_channels=64, hidden_channels=c_hidden, out_channels=64, num_layers=1, edge_dim=86, heads=3)
        self.head1 = nn.Sequential(nn.Dropout(dp_rate_linear), nn.Linear(64, 64), nn.ReLU())
        self.bnorm1=BatchNorm(in_channels=256)
        self.bnorm2=BatchNorm(in_channels=32)

      #  self.head2 = nn.Sequential(nn.Dropout(dp_rate_linear), nn.Linear(c_hidden, c_out), nn.ReLU())
    def forward(self, x, edge_index, edge_attr, batch_idx):
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            batch_idx: Index of batch element for each node
        """

        z=self.GNN2(x, edge_index, edge_attr)
        z= F.relu(x)
        z=self.drop2(x)
        x=x+z



        return x





class Edge_Classify(nn.Module):
    def __init__(self, c_in, c_out, dp_rate_linear=0.3):
        """
        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of output features (usually number of classes)
            dp_rate_linear: Dropout rate before the linear layer (usually much higher than inside the GNN)
            kwargs: Additional arguments for the GNNModel object
        """
        super().__init__()
        c_hidden=512
        self.head1= nn.Sequential( nn.Linear(c_in, c_hidden), nn.ReLU())
        self.head2 = nn.Sequential(nn.Dropout(dp_rate_linear), nn.Linear(c_hidden, c_out))
    def forward(self, x, edge_index, batch_idx):
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            batch_idx: Index of batch element for each node
        """
        src, dst = edge_index

        x_out= torch.cat((x[src], x[dst]), -1)

        x_out=self.head1(x_out)
        x_out=self.head2(x_out)

        return x_out



class LightningGCN(pl.LightningModule):
    def __init__(self, **kwargs):
        super(LightningGCN, self).__init__()

        self.num_features = 1
        self.num_classes = 86
        self.hidden = 256
        self.model = GraphRGCNodel(c_in=256 )
        self.loss_module = nn.CrossEntropyLoss()
        self.final_classifier = Edge_Classify(c_in=512, c_out=86)


    def forward(self, data, mode="train"):
        x, edge_index, edge_attr, batch_index = data.x, data.edge_index, data.edge_attr, data.batch
        edge_attr=edge_attr-1
        x = self.model(x, edge_index,edge_attr,  batch_index)
        x_out= self.final_classifier(x, edge_index,  batch_index)
        loss = self.loss_module(x_out, edge_attr)
        acc = (x_out.argmax(dim=-1) == edge_attr).sum().float() / x_out.shape[0]
        return x, loss, acc


    def training_step(self, batch):

        _, loss, acc= self.forward(batch, mode='train')
        self.log("loss/train", loss, prog_bar=True)
        self.log("accuracy/train", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        #x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        _, loss, acc= self.forward(batch, mode='val')
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),betas=(0.9, 0.999),  lr = 1e-3)


    def test_step(self, batch, batch_idx):
       return