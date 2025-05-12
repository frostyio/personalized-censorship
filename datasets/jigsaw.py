from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd

class ToxicCommentsDataset(Dataset):
    def __init__(self, inputs, labels, tokenizer, device="cpu", max_length=128):
        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        text = self.inputs[idx]
        label = self.labels[idx]

        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        tokenized = {
            key: val.squeeze(0).to(self.device) for key, val in tokenized.items()
        }

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "label": torch.tensor(label, dtype=torch.long).to(self.device),
        }


toxic_columns = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def GetInputAndLabels(type="train", n=None, toxicity_level=None):
    inputs = []
    labels = []
    length = 0

    if type == "train":
        train_df = pd.read_csv("../datasets/jigsaw/train.csv")
        length = len(train_df)
        toxic_inputs = []
        toxic_labels = []
        non_toxic_inputs = []
        non_toxic_labels = []

        for _, row in train_df.iterrows():
            text = row["comment_text"]
            toxicity = row[toxic_columns]
            toxic_sum = toxicity.sum()

            if toxicity_level is not None:
                is_toxic = toxic_sum >= toxicity_level
            else:
                is_toxic = toxicity.any()

            if is_toxic:
                toxic_inputs.append(text)
                toxic_labels.append(1)
            elif toxic_sum == 0:
                non_toxic_inputs.append(text)
                non_toxic_labels.append(0)

            if (
                n is not None
                and len(toxic_inputs) >= n // 2
                and len(non_toxic_inputs) >= n // 2
            ):
                break

        inputs = toxic_inputs[: n // 2] + non_toxic_inputs[: n // 2]
        labels = toxic_labels[: n // 2] + non_toxic_labels[: n // 2]

    elif type == "test":
        test_df = pd.read_csv("../datasets/jigsaw/test.csv")
        test_labels_df = pd.read_csv("../datasets/jigsaw/test_labels.csv")
        length = len(test_df)

        toxic_mask = (
            test_labels_df[toxic_columns].sum(axis=1) >= toxicity_level
            if toxicity_level is not None
            else test_labels_df[toxic_columns].any(axis=1)
        )
        toxic_inputs = test_df[toxic_mask]["comment_text"].tolist()
        toxic_labels = [1] * len(toxic_inputs)

        non_toxic_mask = test_labels_df[toxic_columns].sum(axis=1) == 0
        non_toxic_inputs = test_df[non_toxic_mask]["comment_text"].tolist()
        non_toxic_labels = [0] * len(non_toxic_inputs)

        inputs = toxic_inputs[: n // 2] + non_toxic_inputs[: n // 2]
        labels = toxic_labels[: n // 2] + non_toxic_labels[: n // 2]

    return inputs, labels, length


def GetInputAndLabelsByClass(type="train", n=None, default=False):
    category_data = {col: [] for col in toxic_columns}
    category_data["non_toxic"] = []

    if type == "train":
        df = pd.read_csv("../datasets/jigsaw/train.csv").head(n)
    elif type == "test":
        df = pd.read_csv("../datasets/jigsaw/test.csv").head(n)
        test_labels_df = pd.read_csv("../datasets/jigsaw/test_labels.csv").head(n)
        df = df.merge(test_labels_df, on="id", suffixes=("", "_label"))

    for _, row in df.iterrows():
        text = row["comment_text"]
        matched_categories = [col for col in toxic_columns if row[col] == 1]

        if default:
            for col in matched_categories:
                category_data[col].append(text)
        elif len(matched_categories) == 1:
            category_data[matched_categories[0]].append(text)
        elif len(matched_categories) == 0:
            category_data["non_toxic"].append(text)

    return category_data


def GetDataLoader(tokenizer, batch_size=16, device="cpu", n=None):
    inputs, labels, length = GetInputAndLabels(n=n)
    dataset = ToxicCommentsDataset(inputs, labels, tokenizer, device=device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), length


def GetTestDataLoader(tokenizer, batch_size=16, device="cpu", n=None):
    inputs, labels, length = GetInputAndLabels(type="test", n=n)
    dataset = ToxicCommentsDataset(inputs, labels, tokenizer, device=device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), length
