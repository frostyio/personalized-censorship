from torch import optim, no_grad, argmax, save, load
import torch.nn as nn
from numpy import mean
from tqdm import tqdm
import csv

class Solver:
    def __init__(
        self,
        model,
        n_way=2,
        lr=0.0001,
        device="cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.classifier = nn.NLLLoss()
        self.device = device
        self.n_way = n_way

    def train(self, loader, n_epochs=200, output_file=None):
        device = self.device
        total_steps = n_epochs * len(loader)
        progress_bar = tqdm(total=total_steps, desc="Training", unit="step")
        epoch_losses = []

        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0.0

            for (
                support_embeddings,
                support_labels,
                query_embeddings,
                query_labels,
            ) in loader:
                support_embeddings = support_embeddings.to(device).squeeze(0)
                support_labels = support_labels.to(device).squeeze(0)
                query_embeddings = query_embeddings.to(device).squeeze(0)
                query_labels = query_labels.to(device).squeeze(0)

                self.optimizer.zero_grad()
                log_probs = self.model(
                    support_embeddings, support_labels, query_embeddings, self.n_way
                )

                loss = self.classifier(log_probs, query_labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                progress_bar.update(1)

            avg_loss = epoch_loss / len(loader)
            epoch_losses.append((epoch + 1, avg_loss))
            progress_bar.set_postfix(epoch=epoch + 1, loss=epoch_loss / len(loader))

        progress_bar.close()

        if output_file != None:
            with open(output_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Epoch", "Loss"])
                writer.writerows(epoch_losses)
            print(f"losses saved to {output_file}")

    def evaluate(self, loader):
        device = self.device
        accuracies = []

        self.model.eval()
        with no_grad():
            for _, (
                support_embeddings,
                support_labels,
                query_embeddings,
                query_labels,
            ) in enumerate(loader):
                support_embeddings = support_embeddings.squeeze(0).to(device)
                support_labels = support_labels.squeeze(0).to(device)
                query_embeddings = query_embeddings.squeeze(0).to(device)
                query_labels = query_labels.squeeze(0).to(device)

                log_probs = self.model(
                    support_embeddings,
                    support_labels,
                    query_embeddings,
                    n_way=self.n_way,
                )
                pred_labels = argmax(log_probs, dim=-1)
                accuracies.append((pred_labels == query_labels).float().mean().item())

        print(f"avg test accuracy: {mean(accuracies) * 100:.2f}%")

    def predict(self, support_embeddings, labels, query_embeddings):
        self.model.eval()
        with no_grad():
            log_probs = self.model(
                support_embeddings, labels, query_embeddings, n_way=self.n_way
            )
            pred_labels = argmax(log_probs, dim=-1)
        return pred_labels

    def load(self, path="../models/proto_net.pth"):
        checkpoint = load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer and "optimizer_state_dict" in checkpoint:
            if checkpoint["optimizer_state_dict"] is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print("loaded")

    def save(self, optimizers=False, path="../models/proto_net.pth"):
        save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": (
                    self.optimizer.state_dict() if optimizers else None
                ),
            },
            path,
        )
        print("saved")
