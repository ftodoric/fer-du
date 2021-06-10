import csv

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

# PATHS
TRAIN_DATASET_PATH = "c:/workspace/fer-dl/lab03/resources/sst_train_raw.csv"
VALID_DATASET_PATH = "c:/workspace/fer-dl/lab03/resources/sst_valid_raw.csv"
TEST_DATASET_PATH = "c:/workspace/fer-dl/lab03/resources/sst_test_raw.csv"

VECTOR_REPR_PATH = "c:/workspace/fer-dl/lab03/resources/sst_glove_6b_300d.txt"


@dataclass
class Instance:
    __instance_text: list
    __instance_label: str

    def __init__(self, text_and_label: str):
        text_and_label = text_and_label.strip().split(", ")
        self.__instance_text = text_and_label[0].split()
        self.__instance_label = text_and_label[1]

    def get_text(self) -> list:
        return self.__instance_text

    def get_label(self) -> str:
        return self.__instance_label

    def __repr__(self) -> tuple:
        return repr((self.__instance_text, self.__instance_label))


class Vocab:
    def __init__(self, frequencies: dict = None, labels: bool = False, max_size: int = -1, min_freq: int = 1):
        self.max_size = max_size
        self.min_freq = min_freq

        # Sort words by frequencies
        frequencies = dict(
            sorted(frequencies.items(), key=lambda item: item[1], reverse=True))

        # stoi and itos dictionnaries
        self.stoi = dict()
        self.itos = dict()

        # Determine vocabulary size
        if not labels:
            vocab_size = len(frequencies) + 2 if max_size == -1 else max_size

            # Add special tokens <PAD> & <UNK>
            self.stoi["<PAD>"] = 0
            self.itos[0] = "<PAD>"
            self.stoi["<UNK>"] = 1
            self.itos[1] = "<UNK>"

        else:
            vocab_size = len(frequencies) if max_size == -1 else max_size

        # Create stoi and itos
        i = 2 if not labels else 0
        for token in frequencies:
            if i >= vocab_size:
                break
            if frequencies[token] >= min_freq:
                self.stoi[token] = i
                self.itos[i] = token
                i += 1

    def encode(self, sentence: str or list):
        encoded = []
        if isinstance(sentence, str):
            for token in sentence.split():
                encoded.append(self.stoi[token])
        else:
            for token in sentence:
                try:
                    encoded.append(self.stoi[token])
                except KeyError:
                    encoded.append(self.stoi["<UNK>"])

        return torch.LongTensor(encoded)

    def decode(self, tensor: torch.LongTensor):
        decoded = []
        for index in tensor.tolist():
            decoded.append(self.itos[index])
        return decoded


class NLPDataset(torch.utils.data.Dataset):

    def __init__(self, x_vocab: Vocab, y_vocab: Vocab, instances: List[Instance]):
        self.x_vocab = x_vocab
        self.y_vocab = y_vocab
        self._instances = instances

    @classmethod
    def from_file(cls, csv_file_path):
        frequencies = getFrequencies(TRAIN_DATASET_PATH)
        labelFrequencies = getLabelFrequencies(TRAIN_DATASET_PATH)
        x_vocab = Vocab(frequencies, max_size=-1, min_freq=1)
        y_vocab = Vocab(labelFrequencies, labels=True)

        # Collect instances
        instances = get_instances(csv_file_path)

        # Transform to encoded data with vocabularies
        transformed = []
        for instance in instances:
            transformed.append((x_vocab.encode(instance.get_text()),
                                y_vocab.encode(instance.get_label())))
        return transformed

    @property
    def instances(self):
        tuple_list = list()
        for instance in self._instances:
            tuple_list.append((instance.get_text(), instance.get_label()))
        return tuple_list

    def __getitem__(self, index):
        return (self.x_vocab.encode(self._instances[index].get_text()),
                self.y_vocab.encode(self._instances[index].get_label()))


# UTILITY FUNCTIONS
def getFrequencies(train_dataset_path: str) -> dict:
    with open(train_dataset_path) as csv_file:
        csvr = csv.reader(csv_file)

        frequencies = {}
        for row in csvr:
            for word in row[0].split():
                try:
                    frequencies[word] += 1
                except KeyError:
                    frequencies[word] = 1

    return frequencies


def get_instances(dataset: str) -> List[Instance]:
    instances = list()
    with open(dataset) as file:
        for line in file:
            instances.append(Instance(line))
    return instances


def getLabelFrequencies(train_dataset_path: str) -> dict:
    with open(train_dataset_path) as csv_file:
        csvr = csv.reader(csv_file)

        frequencies = {}
        for row in csvr:
            try:
                frequencies[row[-1].strip()] += 1
            except KeyError:
                frequencies[row[-1].strip()] = 1

    return frequencies


def generateEmbeddingMatrix(vocabulary: Vocab, path: str = None) -> torch.FloatTensor:
    embedding = []
    for token in vocabulary.stoi:
        embedding.append(np.random.normal(0, 1, 300))

    if path is not None:
        # Load file
        with open(path, "r") as vector_repr:
            lines = vector_repr.readlines()

        # Create word-vector dict
        vectors = dict()
        for line in lines:
            line = line.split()
            token = line[0]
            vector = np.array([float(el) for el in line[1:]])
            vectors[token] = vector

        # Assign vectors to words in vocabulary
        for token in vocabulary.stoi:
            try:
                embedding[vocabulary.stoi[token]] = vectors[token]
            except KeyError:
                pass

    # for <PAD> null-vector
    embedding[0] = np.zeros(300)
    embedding[1] = np.array([1.] * 300)

    return torch.FloatTensor(embedding)


def pad_collate_fn(batch: list, padding_index: int = 0):
    text, labels = zip(*batch)

    return (torch.nn.utils.rnn.pad_sequence(text, batch_first=True, padding_value=padding_index),
            torch.tensor(labels),
            torch.tensor([len(element) for element in text]))


def main():
    # Frequencies
    frequencies = getFrequencies(TRAIN_DATASET_PATH)
    labelFrequencies = getLabelFrequencies(TRAIN_DATASET_PATH)

    # Create vocabularies
    x_vocab = Vocab(frequencies, max_size=-1, min_freq=1)
    y_vocab = Vocab(labelFrequencies, labels=True, max_size=-1, min_freq=1)

    # Dataset
    train_dataset = NLPDataset(x_vocab, y_vocab,
                               get_instances(TRAIN_DATASET_PATH))
    instance_text, instance_label = train_dataset.instances[3]
    """ print(f"Text: {instance_text}")
    print(f"Label: {instance_label}")
    print(f"Numericalized text: {x_vocab.encode(instance_text)}")
    print(f"Numericalized label: {y_vocab.encode(instance_label)}") """

    # Embedding matrix
    embedding = generateEmbeddingMatrix(x_vocab, VECTOR_REPR_PATH)
    # print(embedding)
    embedding = torch.nn.Embedding.from_pretrained(embedding, padding_idx=0)

    # __getitem__ implementation
    instance_text, instance_label = train_dataset.instances[3]
    numericalized_text, numericalized_label = train_dataset[3]
    """ print(f"Text: {instance_text}")
    print(f"Label: {instance_label}")
    print(f"Numericalized text: {numericalized_text}")
    print(f"Numericalized label: {numericalized_label}") """

    # DataLoader test
    batch_size = 2      # Only for demonstrative purposes
    shuffle = False     # Only for demonstrative purposes
    train_dataset = NLPDataset.from_file(TRAIN_DATASET_PATH)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                    shuffle=shuffle, collate_fn=pad_collate_fn)
    texts, labels, lengths = next(iter(train_data_loader))
    """ print(f"Texts: {texts}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}") """


if __name__ == "__main__":
    main()
