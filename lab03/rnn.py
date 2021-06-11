import random
import json

import numpy as np
import torch

# Custom imports
import data
from model_eval import evaluate


class LSTM(torch.nn.Module):
    def __init__(self, embedding: torch.FloatTensor):
        super().__init__()

        # Embedding wrapper
        self.__embedding = torch.nn.Embedding.from_pretrained(
            embedding, freeze=True, padding_idx=0)

        # RNN layers
        self.__rnn1 = torch.nn.LSTM(300, 150,
                                    num_layers=2, batch_first=False)
        self.__rnn2 = torch.nn.LSTM(300, 150,
                                    num_layers=2, batch_first=False)

        # FC layers
        self.__fc1 = torch.nn.Linear(150, 150)
        self.__fc2 = torch.nn.Linear(150, 1)

    def all_params(self):
        params = []

        params.extend(self.__rnn1.parameters())
        params.extend(self.__rnn2.parameters())
        params.extend(self.__fc1.parameters())
        params.extend(self.__fc2.parameters())
        params.extend(self.__embedding.parameters())

        return params

    def forward(self, x):
        x = self.__embedding(x)
        x = torch.transpose(x, 0, 1)

        # Consists of (h, c)
        hidden = None

        y, hidden = self.__rnn1(x, hidden)
        y, hidden = self.__rnn2(x, hidden)

        # Last output
        y = y[-1]

        # Linear layer
        y = self.__fc1(y)
        y = torch.relu(y)

        return self.__fc2(y)

    def predict(self, x):
        with torch.no_grad():
            y = torch.sigmoid(self.forward(x))
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
    x_vocab = data.Vocab(
        frequencies, max_size=hyperparameters["max_size"], min_freq=hyperparameters["min_freq"])
    y_vocab = data.Vocab(labelFrequencies, labels=True)

    # Datasets
    train_dataset = data.NLPDataset.from_file(data.TRAIN_DATASET_PATH)
    valid_dataset = data.NLPDataset.from_file(data.VALID_DATASET_PATH)
    test_dataset = data.NLPDataset.from_file(data.TEST_DATASET_PATH)

    # Embedding matrix
    embedding = data.generateEmbeddingMatrix(
        x_vocab, data.VECTOR_REPR_PATH)

    # Baseline model
    lstm = LSTM(embedding)

    optimizer = torch.optim.Adam(
        lstm.all_params(), lr=hyperparameters["learning_rate"])
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
            train_loss = train(lstm, dataloader,
                               optimizer, criterion)
            statistics[seed]["train_loss"] = train_loss

            dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=hyperparameters["valid_batch_size"],
                                                     shuffle=False, collate_fn=data.pad_collate_fn)
            print("\tValidating...")
            valid_evals = evaluate(lstm, dataloader, criterion)
            statistics[seed]["valid"].append(valid_evals)

        # Test dataset
        dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32,
                                                 shuffle=False, collate_fn=data.pad_collate_fn)
        print("Testing...")
        test_evals = evaluate(lstm, dataloader, criterion)
        statistics[seed]["test"] = test_evals

    print("\nAll done.")

    # Write to statistics file
    with open("c:/workspace/fer-dl/lab03/stats/lstm_stats.json", "w") as stats_file:
        stats_file.write(json.dumps(statistics))
