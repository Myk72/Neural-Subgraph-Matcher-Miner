# Query mode for subgraph matching.
import argparse
import torch
import torch.nn.functional as F
import networkx as nx
import torch_geometric.utils as geo_utils
from common import models
from common import utils
from subgraph_matching.config import parse_encoder

def loadM(model_path, args):
    model = models.OrderEmbedder(1, args.hidden_dim, args)
    try:
        state = torch.load(model_path, map_location=utils.get_device())
        model.load_state_dict(state)
        print("debug checkpoint for loadin")
    except Exception as e:
        print("failed", e)
        exit(1)

    model.to(utils.get_device())
    model.eval()
    return model

def graphTodata(graph):
    data = geo_utils.from_networkx(graph)
    if not hasattr(data, 'x') or data.x is None:
        data.x = torch.ones((data.num_nodes, 1))
    data.node_feature = data.x
    return data

def query_matcher(query, target, model):
    Q = graphTodata(query).to(utils.get_device())
    T = graphTodata(target).to(utils.get_device())
    emb_q = model.emb_model(Q)
    emb_t = model.emb_model(T)
    
    score = model(emb_q, emb_t)
    if isinstance(score, tuple):
        score = score[0]
    prob = torch.sigmoid(score).mean().item()
    return prob > 0.5, prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Matcher")
    parse_encoder(parser)
    args = parser.parse_args()
    
    if not getattr(args, "model_path", None):
        args.model_path = "ckpt/model_new.pt"


    # This is test data, it woon't work since the model is trained with enzymes dataset
    target = nx.cycle_graph(6)
    query = nx.complete_graph(4)

    model = loadM(args.model_path, args)
    result, score = query_matcher(query, target, model)
    # print("score", score)

    print(result)