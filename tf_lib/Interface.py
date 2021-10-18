import tensorflow as tf
import tf2onnx.convert
import onnx
import numpy as np


class Interface:
    def __init__(self, net) -> None:
        self.net = net

    def train_net(self, dataset_batches, epochs, verbose=False, **kwargs):
        optimizer = kwargs.get('optimizer', 'adam')
        loss = kwargs.get(
            'loss',
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        metrics = kwargs.get('metrics', ['accuracy'])
        self.net.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        for epoch in range(epochs):
            if verbose:
                print('epoch:', epoch)
            for batch in dataset_batches:
                X, Y = batch
                self.net.fit(X, Y, **kwargs)

    def save_weights(self, filepath, **kwargs):
        return self.net.save_weights(filepath, **kwargs)

    def load_weights(self, filepath, **kwargs):
        return self.net.load_weights(filepath, **kwargs)

    def predict_net(self, X, *args, **kwargs):
        return self.net.predict(X, *args, **kwargs)

    def eval_acc_net(self, X, Y):
        Y_pred = self.net.predict(X)
        Y_pred = np.argmax(Y_pred, axis=1)

        N = Y.shape[0]
        correct = 0
        for i in range(N):
            if Y_pred[i] == Y[i]:
                correct += 1

        return correct/N, N

    def convert2onnx(self, filepath):
        onnx_model, _ = tf2onnx.convert.from_keras(self.net)
        onnx.save(onnx_model, filepath)
