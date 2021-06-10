import random
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Custom imports
import data
from model_eval import evaluate


class BaselineModel(torch.nn.Module):
    def __init__(self, embedding: torch.FloatTensor):
        super().__init__()

        # Embedding wrapper
        self.__embedding = torch.nn.Embedding.from_pretrained(
            embedding, freeze=True, padding_idx=0)

        # Full connected layers
        self.__fc1 = torch.nn.Linear(300, 150)
        self.__fc2 = torch.nn.Linear(150, 150)
        self.__fc3 = torch.nn.Linear(150, 1)

    def forward(self, x) -> torch.Tensor:
        # Get representation vector via embedding
        y = self.__embedding(x)

        # Pooling
        y = torch.mean(y, dim=1)

        # Go through layers
        y = self.__fc1(y)
        y = torch.relu(y)
        y = self.__fc2(y)
        y = torch.relu(y)
        return self.__fc3(y)

    def all_params(self):
        params = []

        params.extend(self.__fc1.parameters())
        params.extend(self.__fc2.parameters())
        params.extend(self.__fc3.parameters())
        params.extend(self.__embedding.parameters())

        return params

    def predict(self, x):
        # With no gradient accumulation
        with torch.no_grad():
            y = self.forward(x)
            y = torch.sigmoid(y)
            y = y.round().int().squeeze(-1)

        return y


def train(model: torch.nn.Module, data,
          optimizer, criterion):
    # Set state for training
    model.train()

    # Go through batches
    losses = list()
    for batch_num, batch in enumerate(data):
        model.zero_grad()

        # Calculate loss
        logits = model.forward(batch[0]).squeeze(-1)
        y = batch[1].float()
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.all_params(), 0.25)
        optimizer.step()

        losses.append(float(loss))

    # At the end of an epoch print loss
    #print(f"loss = {np.mean(losses)}")

    return np.mean(losses)


if __name__ == "__main__":
    # Statistics
    hyperparameters = dict()
    hyperparameters["max_size"] = -1
    hyperparameters["min_freq"] = 1
    hyperparameters["train_batch_size"] = 10
    hyperparameters["valid_batch_size"] = 32
    hyperparameters["test_batch_size"] = 32
    hyperparameters["learning_rate"] = 1e-4
    statistics = dict()
    statistics["hyperparameters"] = hyperparameters

    # Frequencies
    frequencies = data.getFrequencies(data.TRAIN_DATASET_PATH)
    labelFrequencies = data.getLabelFrequencies(data.TRAIN_DATASET_PATH)

    # Vocabs
    x_vocab = data.Vocab(frequencies,
                         max_size=hyperparameters["max_size"],
                         min_freq=hyperparameters["min_freq"])
    y_vocab = data.Vocab(labelFrequencies, labels=True)

    # Datasets
    train_dataset = data.NLPDataset.from_file(data.TRAIN_DATASET_PATH)
    valid_dataset = data.NLPDataset.from_file(data.VALID_DATASET_PATH)
    test_dataset = data.NLPDataset.from_file(data.TEST_DATASET_PATH)

    # Embedding matrix
    embedding = data.generateEmbeddingMatrix(x_vocab,
                                             data.VECTOR_REPR_PATH)

    # Baseline model
    bl_model = BaselineModel(embedding)

    optimizer = torch.optim.Adam(bl_model.all_params(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    iters = 5
    epochs = 5
    for i in range(iters):
        print(f"RUN {i+1}")

        # Set seed
        seed = random.randint(0, 7052020)
        np.random.seed(seed)
        torch.manual_seed(seed)

        statistics[seed] = dict()
        statistics[seed]["train_loss"] = None
        statistics[seed]["valid"] = list()

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}:")

            dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=hyperparameters["train_batch_size"],
                                                     shuffle=True, collate_fn=data.pad_collate_fn)
            print("\tTraining...")
            train_loss = train(bl_model, dataloader,
                               optimizer, criterion)
            statistics[seed]["train_loss"] = train_loss

            dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=hyperparameters["valid_batch_size"],
                                                     shuffle=False, collate_fn=data.pad_collate_fn)
            print("\tValidating...")
            valid_evals = evaluate(bl_model, dataloader, criterion)
            statistics[seed]["valid"].append(valid_evals)

        # Test dataset
        dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=hyperparameters["test_batch_size"],
                                                 shuffle=False, collate_fn=data.pad_collate_fn)
        print("Testing...")
        test_evals = evaluate(bl_model, dataloader, criterion)
        statistics[seed]["test"] = test_evals

    print("\nAll done.")

    # Write to statistics file
    with open("c:/workspace/fer-dl/lab03/stats/baseline_stats.json", "w") as stats_file:
        stats_file.write(json.dumps(statistics))
