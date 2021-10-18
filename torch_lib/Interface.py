import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import numpy as np


class Interface:
    def __init__(self, net) -> None:
        self.net = net

    def train_net(self, dataset_batches, epochs, verbose=False, **kwargs):
        criterion = kwargs.get('criterion', nn.CrossEntropyLoss())
        optimizer = kwargs.get('optimizer', optim.SGD(
            self.net.parameters(),
            lr=kwargs.get('lr', 0.001),
            momentum=kwargs.get('momentum', 0.9)
            ))
        i = 0
        for epoch in range(epochs):
            running_loss = 0.0
            if verbose:
                print('epoch:', epoch)
            for batch in dataset_batches:
                X, Y = batch
                optimizer.zero_grad()
                Y_pred = self.net(X)
                # print(Y_pred, Y)
                loss = criterion(Y_pred, Y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
                i += 1

    def save_weights(self, filepath, **kwargs):
        torch.save(self.net.state_dict(), filepath)

    def load_weights(self, filepath, **kwargs):
        self.net.load_state_dict(torch.load(filepath))

    def predict_net(self, X, *args, **kwargs):
        return self.net(X)

    def eval_acc_net(self, X, Y):
        Y_pred = self.net(X).detach().numpy()
        Y_pred = np.argmax(Y_pred, axis=1)

        N = Y.shape[0]
        correct = 0
        for i in range(N):
            if Y_pred[i] == Y[i]:
                correct += 1

        return correct/N, N

    def convert2onnx(self, filepath, dummy_input):
        torch.onnx.export(self.net, dummy_input, filepath, verbose=False)
