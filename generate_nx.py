# Converts CSV adj and features files (outputs of generate.py)
# to NetworkX graphs
# Looks recurisively for *_adj.csv files and converts them to
# graphs, and pickles them to the output file.

from pathlib import Path
import pickle
import networkx as nx
import pandas as pd

dir_name = "graphs/"


def convert_csv_to_nx(path):
    # print("Processing", path)
    g = nx.read_adjlist(path, delimiter=",", nodetype=str, encoding="utf-8")

    path_features = str(path).replace("adj", "features")
    df = pd.read_csv(
        path_features,
        index_col=False,
        header=None,
        names=["id", "generic_type", "specific_type", "value"],
    )
    # df.fillna("", inplace=True)
    del df["value"]
    df.set_index("id", inplace=True)
    attrs = df.to_dict(orient="index")

    nx.set_node_attributes(g, attrs)

    return g


if __name__ == "__main__":
    graphs = []
    for path in Path(dir_name).rglob("*_adj.csv"):
        graphs.append(convert_csv_to_nx(path))

    print("Finished", len(graphs))
    with open("graphs.pkl", "wb") as f_out:
        pickle.dump(graphs, f_out)
        print("saved!")
