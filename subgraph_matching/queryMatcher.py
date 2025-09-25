# Query mode for subgraph matching
import argparse
import torch
import torch_geometric.utils as pyg_utils
import networkx as nx
from common import models, utils
from subgraph_matching.config import parse_encoder
from visualizer import visualizer
import os
import pickle
from .alignment import gen_alignment_matrix
from networkx.algorithms import isomorphism as iso


def loadM(model_path, args):
    model = models.OrderEmbedder(1, args.hidden_dim, args)
    stat = torch.load(model_path, map_location=utils.get_device())
    model.load_state_dict(stat)
    model.to(utils.get_device())
    model.eval()
    return model

def graphTodata(graph):
    data = pyg_utils.from_networkx(graph)
    if not hasattr(data, 'x') or data.x is None:
        data.x = torch.ones((data.num_nodes, 1))
    data.node_feature = data.x
    return data.to(utils.get_device())

def query_matcher(query, target, model, method_type="order", mode="hybrid"):
    Q = graphTodata(query)
    T = graphTodata(target)

    emb_q = model.emb_model(Q)
    emb_t = model.emb_model(T)
    

    scor = model(emb_q, emb_t)
    score = scor[0] if isinstance(scor, tuple) else scor
    orderEmbProb = torch.sigmoid(score).mean().item()
    print("ord score:", orderEmbProb)
    alignScore = gen_alignment_matrix(model, query, target, method_type=method_type).max().item()
    alignScore = torch.sigmoid(torch.tensor(alignScore)).item()
    print("align mat:", alignScore)

    print("built",iso.GraphMatcher(target, query).subgraph_is_isomorphic())
    if mode == "order":
        return orderEmbProb > 0.5, orderEmbProb
    elif mode == "align":
        return alignScore > 0.5, alignScore
    
    a = 0.6
    res = (a * orderEmbProb + (1 - a) * alignScore)
    return res > 0.5, res



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Matcher")
    parse_encoder(parser)

    # Just like alignment.py formt
    parser.add_argument("--query_path", type=str, default="", help="Pickle file for query graph")
    parser.add_argument("--target_path", type=str, default="", help="Pickle file for target graph")
    args = parser.parse_args()

    if not getattr(args, "model_path", None):
        args.model_path = "ckpt/model.pt"
    model = loadM(args.model_path, args)

        
    if args.target_path and os.path.exists(args.target_path):
        with open(args.target_path, "rb") as f:
            target = pickle.load(f)
    else:
        target = nx.gnp_random_graph(15, 0.3)
        
    if args.query_path and os.path.exists(args.query_path):
        print(args.query_path)
        with open(args.query_path, "rb") as f:
            query = pickle.load(f)
    else:
        sample_nodes = list(target.nodes)[:6]

        # pos ex
        # query = target.subgraph(sample_nodes).copy()

        # neg ex
        target_nodes = list(target.nodes)
        max_node = max(target_nodes)
        query = nx.gnp_random_graph(3, 0.4)
        query = nx.relabel_nodes(query, lambda x: x + max_node + 1)



    result, score = query_matcher(query, target, model)
    print("res:", result)
    print("score:", score)
    

    visualizer.visualize_pattern_graph_ext(query, args, {})
    visualizer.visualize_pattern_graph_ext(target, args, {})
