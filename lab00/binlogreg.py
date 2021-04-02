import numpy as np


# stable softmax
def stable_softmax(x):
    exp_x_shifted = np.exp(x - np.max(x))
    probs = exp_x_shifted / np.sum(exp_x_shifted)
    return probs


def sigma_P(y, s):
    # binary classification - classes: 0 and 1
    return np.exp(s) ** y / (1 + np.exp(s))


def binlogreg_train(X, Y_):
    """
    Args
        X: learning dataset, np.array NxD
        Y_: class indexes, np.array Nx1

    Return
        w, b: log reg parameters
    """

    param_niter = 10
    param_delta = 0.1

    w = np.random.randn(X.shape[1], 1)
    b = np.zeros((1, X.shape[0]))

    # grad descent (param_niter number of iterations)
    for i in range(param_niter):
        # classification measures - Nx1
        scores = np.dot(X, w) + b

        # class probabilities c_1 - Nx1
        probs = stable_softmax(scores)

        # loss - scalar
        loss = np.sum(-np.log(sigma_P(Y_, scores))) * 1/len(X)

        # dijagnostički ispis
        if i % 10 == 0:
            pass
        print("iteration {}: loss {}".format(i, loss))

        # loss derivations by scores - Nx1
        iverson = 1 * (probs > 0.5)
        dL_ds = probs - iverson

        # parameters gradients
        grad_w = 1 / len(X) * np.dot(np.transpose(dL_ds), X)
        print(grad_w)
        grad_b = 1 / len(X) * np.transpose(dL_ds)
        print(grad_b)

        # update parameters
        w += -param_delta * np.transpose(grad_w)
        b += -param_delta * np.transpose(grad_b)


# MAIN
X = np.array([[1, 3],
              [3, 2],
              [1, 2]])
Y_ = np.array([[0],
               [1],
               [0]])
binlogreg_train(X, Y_)
