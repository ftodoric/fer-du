import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import skimage as ski
import skimage.io

import torch
from torchvision.datasets import MNIST
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

MAX_EPOCHS = 8
BATCH_SIZE = 50
WEIGHT_DECAY = 1e-2

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out_task3' / \
    'lambda_{:.3f}'.format(WEIGHT_DECAY)


def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]


def draw_conv_filters(epoch, layer, save_dir=SAVE_DIR):
    n_columns = 8
    border = 1

    w = layer.weight.clone().detach().numpy()

    N, C, W, H = w.shape[:4]

    w_T = w.transpose(2, 3, 1, 0)
    w_T -= w_T.min()
    w_T /= w_T.max()

    n_rows = int(np.ceil(N / n_columns))

    width, height = [x * y + (x - 1) * border for x, y in zip([n_columns, n_rows],
                                                              [W, H])]
    image = np.zeros([height, width, C])

    for i in range(N):
        c = int(i % n_columns) * (W + border)
        r = int(i / n_columns) * (H + border)

        image[r: r + H, c: c + W, :] = w_T[:, :, :, i]
        filename = '%s_epoch_%02d.png' % ("conv_1",
                                          epoch)
    ski.io.imsave(os.path.join(save_dir, filename),
                  np.array(image * 255., dtype=np.uint8))


class CNN_L2(nn.Module):
    def __init__(self, in_ch_1, out_ch_1, out_ch_2, out_features_1, class_count):
        super().__init__()

        # 1st convolution layer
        self.conv_1 = nn.Conv2d(in_channels=in_ch_1,
                                out_channels=out_ch_1,
                                kernel_size=5,
                                stride=1, padding=2, bias=True)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2nd convolution layer
        self.conv_2 = nn.Conv2d(in_channels=out_ch_1,
                                out_channels=out_ch_2,
                                kernel_size=5,
                                stride=1, padding=2, bias=True)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # FC layers
        self.fc_1 = nn.Linear(in_features=out_ch_2 * 7 * 7,
                              out_features=out_features_1,
                              bias=True)
        self.fc_logits = nn.Linear(in_features=out_features_1,
                                   out_features=class_count,
                                   bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.fc_logits:
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        self.fc_logits.reset_parameters()

    def forward(self, x):
        x = x.float()

        # 1st convolution layer pass
        h_conv1 = self.conv_1(x)
        h_conv1 = self.maxpool_1(h_conv1)
        h_conv1 = torch.relu(h_conv1)
        # može i h_conv1.relu() ili nn.functional.relu(h_conv1)

        # 2nd convolution layer pass
        h_conv2 = self.conv_2(h_conv1)
        h_conv2 = self.maxpool_2(h_conv2)
        h_conv2 = torch.relu(h_conv2)

        # Flatten output to 1D
        h_conv = h_conv2.view(h_conv2.shape[0], -1)

        # 1st FC layer pass
        h_fc = self.fc_1(h_conv)
        h_fc = torch.relu(h_fc)

        # 2nd FC layer pass
        logits = self.fc_logits(h_fc)
        return logits

    def predict(self, x):
        return torch.argmax(self.forward(x))

    def loss(self, x, y):
        log_sum = torch.log(torch.sum(torch.exp(x), dim=1))
        sum_mul = torch.sum(x * y, dim=1)

        return torch.mean(log_sum - sum_mul)

    def get_val_losses(self, x, y):
        return float(self.loss(self.forward(x).clone().detach(), y))

    def train(self, x, y, x_valid, y_valid, regularization_factor):
        # Make tensors
        x = torch.tensor(x)
        y = torch.tensor(y)
        x_valid = torch.tensor(x_valid)
        y_valid = torch.tensor(y_valid)

        # Set optimizer with weight decay
        optimizer = SGD(self.parameters(), lr=1e-2,
                        weight_decay=regularization_factor)

        # Draw a filter
        draw_conv_filters(0, self.conv_1)

        losses = list()
        validation_losses = list()
        # Iterate through epochs
        for i in range(1, MAX_EPOCHS+1):
            print("Epoch: {}".format(i))

            # Prepare mini batches
            x_batches = torch.split(x, BATCH_SIZE)
            y_batches = torch.split(y, BATCH_SIZE)

            batches_loss = list()
            for (x_batch, y_batch) in zip(x_batches, y_batches):
                # Calculate loss
                loss = self.loss(self.forward(x_batch), y_batch)
                print("Loss: {}".format(float(loss)))

                # Save batch loss
                batches_loss.append(float(loss))

                # Calculate loss gradient and update model parameters
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Save average epoch loss
            losses.append(np.mean(batches_loss))
            validation_losses.append(self.get_val_losses(x_valid, y_valid))

            # Draw a filter
            draw_conv_filters(i, self.conv_1)

        # Plot train and validation set losses
        epochs = [i for i in range(MAX_EPOCHS)]
        plt.plot(epochs,
                 losses,
                 label="train losses")
        plt.plot(epochs,
                 validation_losses,
                 label="validation losses")
        plt.legend(loc="best")
        plt.show()


if __name__ == "__main__":
    # Define CNN model
    model = CNN_L2(in_ch_1=1,
                   out_ch_1=16,
                   out_ch_2=32,
                   out_features_1=512,
                   class_count=10)

    # Prepare MNIST dataset
    ds_train, ds_test = MNIST(DATA_DIR, train=True,
                              download=True), MNIST(DATA_DIR, train=False)
    train_x = ds_train.data.reshape(
        [-1, 1, 28, 28]).numpy().astype(float) / 255
    train_y = ds_train.targets.numpy()
    train_x, valid_x = train_x[:55000], train_x[55000:]
    train_y, valid_y = train_y[:55000], train_y[55000:]
    test_x = ds_test.data.reshape(
        [-1, 1, 28, 28]).numpy().astype(float) / 255
    test_y = ds_test.targets.numpy()
    train_mean = train_x.mean()
    train_x, valid_x, test_x = (
        x - train_mean for x in (train_x, valid_x, test_x))
    train_y, valid_y, test_y = (dense_to_one_hot(y, 10)
                                for y in (train_y, valid_y, test_y))

    # Train model
    model.train(train_x,
                train_y,
                valid_x,
                valid_y,
                WEIGHT_DECAY)
