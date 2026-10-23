"""Las piezas de machine learning del capítulo 0, en NumPy y sin atajos.

Un modelo lineal, el error cuadrático medio, su gradiente, el descenso de
gradiente con minibatches, la estandarización, la densidad y la
log-densidad de una normal, y la sigmoide. El resto del libro (BC, PPO,
GAIL) usa versiones de PyTorch de estas mismas ideas; aquí se escriben a
mano para verlas por dentro.

Las regiones que se incluyen en el libro tienen líneas de 76 caracteres
como máximo, para que quepan en la página.
"""

import numpy as np


# region: linear_model
def predict(w, b, x):
    """Un modelo lineal: cada entrada por su peso, más un sesgo.
    `x` tiene una fila por ejemplo y una columna por entrada."""
    return x @ w + b
# endregion: linear_model


# region: mse
def mse(prediction, target):
    """Error cuadrático medio: el promedio de (predicción − objetivo)²."""
    return float(np.mean((prediction - target) ** 2))
# endregion: mse


# region: mse_gradient
def mse_gradient(w, b, x, y):
    """Cuánto cambia el error al mover cada peso y el sesgo.

    Para cada peso: el promedio de 2 × error × su entrada. El sesgo es un
    peso cuya entrada vale siempre 1."""
    error = predict(w, b, x) - y
    return 2 * x.T @ error / len(y), float(2 * error.mean())
# endregion: mse_gradient


# region: gradient_descent
def gradient_descent(x, y, learning_rate, steps, minibatch_size=None,
                     seed=0):
    """Arranca con todo en cero y da `steps` pasos cuesta abajo.

    Con `minibatch_size`, cada paso mira solo una muestra al azar de ese
    tamaño. Devuelve los pesos, el sesgo y el error tras cada paso."""
    rng = np.random.default_rng(seed)
    w, b = np.zeros(x.shape[1]), 0.0
    history = []
    for _ in range(steps):
        if minibatch_size:
            idx = rng.integers(len(y), size=minibatch_size)
        else:
            idx = slice(None)
        grad_w, grad_b = mse_gradient(w, b, x[idx], y[idx])
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b
        history.append(mse(predict(w, b, x), y))
    return w, b, history
# endregion: gradient_descent


# region: standardize
def standardize(x):
    """Cada columna, con media 0 y desviación estándar 1.

    Devuelve también la media y la desviación, para aplicar la misma
    transformación a datos nuevos."""
    mean, std = x.mean(axis=0), x.std(axis=0)
    return (x - mean) / std, mean, std
# endregion: standardize


def unstandardize_weights(w, b, mean, std):
    """Pasa los pesos aprendidos con entradas estandarizadas a las unidades
    originales."""
    w_original = w / std
    return w_original, float(b - np.sum(w_original * mean))


# region: normal
def normal_density(x, mean, std):
    """La campana de una normal en x. Es una densidad, no una
    probabilidad: puede pasar de 1."""
    z = (x - mean) / std
    return float(np.exp(-0.5 * z**2) / (std * np.sqrt(2 * np.pi)))


def normal_log_density(x, mean, std):
    """El logaritmo de la misma campana, calculado directamente: sin
    exponenciales que se vayan a cero."""
    z = (x - mean) / std
    return float(-0.5 * z**2 - np.log(std) - 0.5 * np.log(2 * np.pi))
# endregion: normal


# region: sigmoid
def sigmoid(z):
    """Aplasta cualquier número al intervalo (0, 1): convierte un puntaje
    en una probabilidad."""
    return float(1 / (1 + np.exp(-z)))
# endregion: sigmoid
