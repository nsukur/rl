import torch
import pickle
import torch_geometric.transforms as T
import sys
import numpy as np
from torch.nn import Linear, Sequential
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv, GlobalAttention
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


def train(model, dataset, optimizer, variational):
    model.train()
    losses = []
    for d in dataset:
        train_data, _, _ = d

        optimizer.zero_grad()

        z = model.encode(train_data.x, train_data.edge_index)
        loss = model.recon_loss(z, train_data.pos_edge_label_index)
        if variational:
            loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    return np.mean(losses)


@torch.no_grad()
def test(model, dataset):
    model.eval()

    auc, ap = [], []

    for d in dataset:
        _, _, data = d
        # print(data.__dict__)
        if "neg_edge_label_index" not in data:
            continue  # small graphs have no enough data for testing
        z = model.encode(data.x, data.edge_index)
        auc_val, ap_val = model.test(
            z, data.pos_edge_label_index, data.neg_edge_label_index
        )
        auc.append(auc_val)
        ap.append(ap_val)

    return np.mean(auc), np.mean(ap)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading nx graphs")
    graphs = pickle.load(open("graphs.pkl", "rb"))

    variational = False
    linear = False
    epochs = 1000

    if "--try" in sys.argv:
        graphs = graphs[:50]
        epochs = 10

    pyg_graphs = []
    print("Converting graphs")

    for g in tqdm(graphs):
        pyg = from_networkx(g)
        pyg_graphs.append(pyg)

    transform = T.Compose(
        [
            T.NormalizeFeatures(),
            T.ToDevice(device),
            T.RandomLinkSplit(
                num_val=0.01,
                num_test=0.1,
                is_undirected=False,
                split_labels=True,
                add_negative_train_samples=False,
            ),
        ]
    )

    for graph in pyg_graphs:
        # node features
        graph.x = (
            torch.stack((graph.generic_type, graph.specific_type), dim=1)
            .float()
            .to(device)
        )
    dataset = transform(pyg_graphs)

    in_channels, out_channels = 2, 128

    if not variational:
        model = GAE(GCNEncoder(in_channels, out_channels))
    else:
        model = VGAE(VariationalGCNEncoder(in_channels, out_channels))

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print("Data loaded")

    for epoch in range(1, epochs + 1):
        loss = train(model, dataset, optimizer, variational)
        auc, ap = test(model, dataset)
        print(f"Epoch: {epoch:03d}, loss: {loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")

    torch.save(model, "gae.pt")
    print("model saved")
