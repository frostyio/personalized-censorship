from transformers import BertForSequenceClassification, get_scheduler, BertTokenizer
from tqdm import tqdm
from torch.optim import AdamW
import torch
from sklearn.metrics import accuracy_score
import os
import csv

class FinetunedBert:
    def __init__(
        self,
        model_name="bert-base-uncased",
        tokenizer_fn=BertTokenizer,
        bert_fn=BertForSequenceClassification,
        device="cpu",
        lr=2e-5,
    ):
        self.model = bert_fn.from_pretrained(model_name, num_labels=2)
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.device = device
        self.bert_fn = bert_fn
        self.tokenizer_fn = tokenizer_fn

    def train(self, loader, epochs=3, output_file=None):
        self.model.to(self.device)
        num_training_steps = epochs * len(loader)
        scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        epoch_losses = []

        self.model.train()

        for epoch in range(epochs):
            loop = tqdm(loader, leave=True)
            epoch_loss = 0

            for batch in loop:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = (
                    outputs.loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                loop.set_description(f"epoch {epoch}")
                loop.set_postfix(loss=loss.item())

                epoch_loss += loss.item()

            epoch_losses.append((epoch, epoch_loss / len(loader)))

        if output_file != None:
            with open(output_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Epoch", "Loss"])
                writer.writerows(epoch_losses)
            print(f"losses saved to {output_file}")

    def accuracy(self, loader):
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        return accuracy_score(true_labels, predictions)

    def load(self, path="../models/fine_tuned_bert"):
        self.model = self.bert_fn.from_pretrained(path)
        self.optimizer.load_state_dict(torch.load(os.path.join(path, "optimizer.pth")))
        tokenizer = self.tokenizer_fn.from_pretrained(path)
        print("loaded")
        return tokenizer

    def save(self, tokenizer, optimizers=False, path="../models/fine_tuned_bert"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "pytorch_model.bin"))
        if optimizers and self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pth"))
        tokenizer.save_pretrained(path)
        print("saved")
