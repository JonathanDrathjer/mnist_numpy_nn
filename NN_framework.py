from itertools import pairwise

import numpy as np


def init_NN(hidden_layers: list[int]):
    if not all(isinstance(num, int) and num > 0 for num in hidden_layers):
        raise ValueError("Layers can only have a positive integer number of neurons.")

    layers = [784] + hidden_layers + [10]

    w = []
    b = []

    for n_in, n_out in pairwise(layers):
        w.append(np.random.randn(n_out, n_in) * np.sqrt(2 / n_in))
        b.append(np.zeros(n_out))

    return w, b


def forward(w, b, image):
    a = image
    for k in range(len(w) - 1):
        z = w[k] @ a + b[k]
        a = np.maximum(0, z)

    z = w[-1] @ a + b[-1]
    z -= np.max(z)
    exp_z = np.exp(z)
    q = exp_z / np.sum(exp_z)

    p = np.argmax(q)
    return q, p


def train(w, b, x_train, y_train, epochs, batch_size, lam, learning_rate):
    N = len(x_train)
    d = len(w)

    for epoch in range(epochs):
        indices = np.random.permutation(N)
        num_batches = int(np.ceil(N / batch_size))
        batches = np.array_split(indices, num_batches)

        for batch in batches:
            x_batch = x_train[batch]
            y_batch = y_train[batch]

            grad_w = [np.zeros_like(W) for W in w]
            grad_b = [np.zeros_like(B) for B in b]

            for x, y in zip(x_batch, y_batch):
                # forward pass
                z = []
                a = [x]

                a_curr = x
                for k in range(d - 1):
                    z_curr = w[k] @ a_curr + b[k]
                    a_curr = np.maximum(0, z_curr)
                    z.append(z_curr)
                    a.append(a_curr)

                z_d = w[-1] @ a_curr + b[-1]
                z_shifted = z_d - np.max(z_d)
                exp_z = np.exp(z_shifted)
                q = exp_z / np.sum(exp_z)
                z.append(z_d)
                a.append(q)

                # backprop
                d_b = [None] * d
                d_w = [None] * d
                delta = q - y
                for k in range(d - 1, -1, -1):
                    d_b[k] = delta
                    d_w[k] = np.outer(delta, a[k])
                    if k > 0:
                        delta = w[k].T @ delta * (z[k - 1] > 0)

                # add gradients to sum
                for k in range(d):
                    grad_w[k] += d_w[k]
                    grad_b[k] += d_b[k]

            actual_batch_size = len(batch)
            for k in range(d):
                # avg grad
                grad_w[k] /= actual_batch_size
                grad_b[k] /= actual_batch_size

                # add regularizer
                grad_w[k] += lam * w[k]

                # take step
                w[k] -= learning_rate * grad_w[k]
                b[k] -= learning_rate * grad_b[k]

        print(epoch, "is done")

    return w, b


def test(w, b, x_test, y_test):
    correct = 0
    for image, label in zip(x_test, y_test):
        _, p = forward(w, b, image)
        if p == label:
            correct += 1

    print("NN has accuracy", 100 * (correct / len(x_test)), "%")
