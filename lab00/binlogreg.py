import numpy as np
import data


# stable softmax
def softmax(s):
    return np.exp(s) / (1 + np.exp(s))


def sigma_P(y, scores):
    # binary classification - classes: 0 and 1
    probs = []
    for i in range(len(y)):
        prob = np.exp(scores[i, 0])**(1 - y[i])/(1 + np.exp(scores[i, 0]))
        probs.append([prob])
    probs = np.array(probs)
    return probs


def binlogreg_train(X, Y_):
    """
    Args
        X: learning dataset, np.array NxD
        Y_: class indexes, np.array Nx1

    Return
        w, b: log reg parameters
    """

    param_niter = 10000
    param_delta = 1e-2

    w = np.random.randn(X.shape[1], 1)
    b = 0

    # grad descent (param_niter number of iterations)
    for i in range(param_niter):
        # classification measures - Nx1
        scores = np.dot(X, w) + b

        # class probabilities c_1 - Nx1
        probs = softmax(scores)

        # loss - scalar
        loss = np.sum(-np.log(sigma_P(Y_, scores))) * 1/len(X)

        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # loss derivations by scores - Nx1
        iverson = 1 * (probs < 0.5)
        dL_ds = probs - iverson

        # parameters gradients
        grad_w = 1 / len(X) * np.dot(np.transpose(dL_ds), X)
        grad_b = 1 / len(X) * np.sum(dL_ds)

        # update parameters
        w -= param_delta * np.transpose(grad_w)
        b -= param_delta * np.transpose(grad_b)

    return w, b


def binlogreg_classify(X, w, b):
    return softmax(np.dot(X, w) + b)


# MAIN
if __name__ == '__main__':
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(2, 100)

    # train the model
    w, b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = np.round(probs[:,0]).astype(int)

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print(accuracy)
    print(recall, precision, AP)

