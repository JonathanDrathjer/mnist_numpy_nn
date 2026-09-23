import numpy as np


def init_NN(hidden_layers: list[int]):
    if not all(isinstance(num, int) and num > 0 for num in hidden_layers):
        raise ValueError("Layers can only have a positive integer number of neurons.")

    layers = [784] + hidden_layers + [10]

    w = []
    b = []

    for n_in, n_out in zip(layers[:-1], layers[1:]):
        w.append(np.zeros((n_out, n_in)))
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

    return q

