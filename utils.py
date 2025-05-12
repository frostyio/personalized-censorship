from torch import no_grad, tensor
from torch.utils.data import Dataset, DataLoader
import random

def EmbedUserInput(tokenizer, bert, inputs, device="cpu"):
    bert.to(device)
    tokenized = tokenizer(
        inputs, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    tokenized = {key: val.to(device) for key, val in tokenized.items()}

    with no_grad():
        outputs = bert(**tokenized, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]
    cls_embedding = last_hidden_state[:, 0, :]
    return cls_embedding


class FewShotEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels, n_way, k_shot, q_queries, device="cpu"):
        self.embeddings = embeddings
        self.labels = tensor(labels).to(device)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        sampled_classes = random.sample(set(self.labels.tolist()), self.n_way)
        support_set = []
        query_set = []
        support_labels = []
        query_labels = []

        for cls in sampled_classes:
            cls_indices = [i for i, label in enumerate(self.labels) if label == cls]
            selected_indices = random.sample(cls_indices, self.k_shot + self.q_queries)
            support_set.extend(selected_indices[: self.k_shot])
            query_set.extend(selected_indices[self.k_shot :])
            support_labels.extend([cls] * self.k_shot)
            query_labels.extend([cls] * self.q_queries)

        support_embeddings = self.embeddings[support_set]
        query_embeddings = self.embeddings[query_set]

        return (
            support_embeddings,
            tensor(support_labels),
            query_embeddings,
            tensor(query_labels),
        )


def GetFewShotDataLoader(embeddings, labels, n_way, k_shot, q_queries, device="cpu"):
    few_shot_dataset = FewShotEmbeddingDataset(
        embeddings, labels, n_way, k_shot, q_queries, device=device
    )
    return DataLoader(few_shot_dataset, batch_size=1, shuffle=True)
