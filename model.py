import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalNetwork(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=128, distance_metric="euclidean"):
        super(PrototypicalNetwork, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.distance_metric = distance_metric

    def compute_prototypes(self, support_embeddings, support_labels, n_way):
        prototypes = []
        for cls in range(n_way):
            mask = support_labels == cls
            if not mask.any():
                raise ValueError(f"no support examples found for class {cls}")
            class_embeddings = support_embeddings[mask]

            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        return torch.stack(prototypes)

    def compute_distances(self, query_embeddings, prototypes):
        if self.distance_metric == "euclidean":
            return torch.cdist(query_embeddings, prototypes, p=2)
        elif self.distance_metric == "cosine":
            query_norm = F.normalize(query_embeddings, p=2, dim=-1)
            proto_norm = F.normalize(prototypes, p=2, dim=-1)
            return 1 - torch.mm(query_norm, proto_norm.t())
        else:
            raise ValueError(f"unsupported distance metric: {self.distance_metric}")

    def forward(self, support_embeddings, support_labels, query_embeddings, n_way):
        support_embeddings = self.head(support_embeddings)
        query_embeddings = self.head(query_embeddings)

        prototypes = self.compute_prototypes(support_embeddings, support_labels, n_way)
        distances = self.compute_distances(query_embeddings, prototypes)

        log_probs = F.log_softmax(-distances, dim=-1)
        return log_probs
