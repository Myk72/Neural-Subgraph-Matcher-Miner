# Query mode for subgraph matching.
import argparse

import torch
import networkx as nx
import torch_geometric.utils as pyg_utils
from common import models, utils
from subgraph_matching.config import parse_encoder
from common.data import DiskDataSource
from visualizer import visualizer

def loadM(model_path, args):
    model = models.OrderEmbedder(1, args.hidden_dim, args)

    state_dict = torch.load(model_path, map_location=utils.get_device())
    model.load_state_dict(state_dict)

    model.to(utils.get_device())
    model.eval()
    return model

def graphTodata(graph):
    data = pyg_utils.from_networkx(graph)
    if not hasattr(data, 'x') or data.x is None:
        data.x = torch.ones((data.num_nodes, 1))
    data.node_feature = data.x
    return data

def query_matcher(query, target, model):
    Q = graphTodata(query).to(utils.get_device())
    T = graphTodata(target).to(utils.get_device())

    emb_q = model.emb_model(Q)
    emb_t = model.emb_model(T)

    scor = model(emb_q, emb_t)
    score = scor[0] if isinstance(scor, tuple) else scor
    prob = torch.sigmoid(score).mean().item()

    return prob > 0.5, prob

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Query Matcher")
    parse_encoder(parser)
    args = parser.parse_args()

    if not getattr(args, "model_path", None):
        args.model_path = "ckpt/model.pt"


    data_source = DiskDataSource("enzymes", node_anchored=args.node_anchored)
    target = data_source.graphs[0]
    query = target.subgraph(list(range(5)))

    model = loadM(args.model_path, args)
    result, score = query_matcher(query, target, model)

    print(f"result {result} (score={score})")
    
    visualizer.visualize_pattern_graph_ext(query, args, {})
    visualizer.visualize_pattern_graph_ext(target, args, {})