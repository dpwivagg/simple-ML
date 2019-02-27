import scipy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_z(x, W):
    z = np.dot(x, W.T)
    return z


def compute_a(z):
    z[z <= 0] = 0
    return z


def compute_softmax(z):
    y_hat = np.exp(z)
    y_hat_sum = np.sum(y_hat, axis=1)
    a = y_hat.T / y_hat_sum
    return a.T


def compute_L_unreg(x, y, W):
    n = y.shape[0]
    z = compute_z(x, W)
    a = compute_a(z)

    cross_loss = np.array([np.dot(np.log(a[i]).T, y[i, :]) for i in range(n)])

    L = -(1 / (2 * n)) * cross_loss.sum()

    return L


def compute_L(x, y, W, alpha):
    n = y.shape[0]
    z = compute_z(x, W)
    a = compute_a(z)

    w_dot_w = np.array([np.dot(W[i], W[i].T) for i in range(W.shape[0] - 1)])
    reg = (alpha / 2) * w_dot_w.sum()
    cross_loss = np.array([np.dot(safe_log(a[i]).T, y[i,:]) for i in range(n)])

    L = -(1 / (2 * n)) * cross_loss.sum() + reg

    return L


def safe_log(arr):
    a = np.array([])
    for i in arr:
        if i == 0:
            a = np.append(a, -1e6)
        else:
            a = np.append(a, np.log(i))

    return a


def compute_gradient_dW(x, y, W):
    n = y.shape[0]
    y_hat = compute_a(compute_z(x, W))
    error = y - y_hat
    dL_dW = -(1 / n) * np.dot(x.T, error)
    return dL_dW.T


def compute_gradient_db(x, y, W):
    n = y.shape[0]
    y_hat = compute_a(compute_z(x, W))
    error = y - y_hat
    dL_db = -(1 / n) * error
    return dL_db.T


def update_W(W, dL_dW, epsilon):
    W = W - epsilon * dL_dW[:,0:-1]
    return W


def update_b(b, dL_db, epsilon):
    b = b - epsilon * dL_db[:,-1]
    return b


def train(X, Y, batch_size, epsilon, n_epochs, alpha):
    # number of features
    p = X.shape[1]
    # number of classes
    c = Y.shape[1]
    # number of instances
    l = X.shape[0]

    # randomly initialize W and b
    # W = np.asmatrix(np.random.rand(c, p))
    W = np.zeros((c, p))

    num_batches = l / batch_size
    if not num_batches.is_integer():
        print("Not a whole number of batches. %.2f batches with batch size %d." % (num_batches, batch_size))
    num_batches = round(num_batches)

    L_vec = np.array([])

    for epoch in range(n_epochs):
        # Create a random order for the samples
        order = np.random.permutation(l)
        pbar = tqdm(range(num_batches), unit="batch")
        for i in pbar:
            elt_indicies = order[i*batch_size:(i*batch_size)+batch_size]
            x = X[elt_indicies]
            # x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
            y = Y[elt_indicies]

            dL_dW = compute_gradient_dW(x, y, W)
            dL_db = compute_gradient_db(x, y, W)
            W = update_W(W[:,0:-1], dL_dW, epsilon)
            b = update_b(W[:,-1], dL_db, epsilon)
            W = np.append(W, np.array([b]).T, axis=1)
            L = compute_L(X, Y, W, alpha)
            L_vec = np.append(L_vec, L)
            acc = compute_acc(X, Y, W)
            pbar.set_description("Loss: %.2f. Accuracy: %.2f. Epoch progress" % (L, acc))

        acc = compute_acc(X, Y, W)
        print("\nEpoch %d completed. Epoch loss: %.2f. Epoch accuracy: %.2f\n" % (epoch+1, L, acc))

    return W, b, L_vec


def compute_acc(X, Y, W):
    y_hat = compute_a(compute_z(X, W))
    y_hat_indicies = np.argmax(y_hat, axis=1)
    y_indicies = np.argmax(Y, axis=1)
    error = np.array([y_indicies == y_hat_indicies])
    acc = error.sum() / X.shape[0]
    return acc





X = np.load('mnist_train_images.npy')
X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
Y = np.load('mnist_train_labels.npy')

W, b, L = train(X, Y, batch_size=75, epsilon=0.005, n_epochs=2, alpha=.005)

""" Performance Evaluation with the Validation Set """
X_valid = np.load('mnist_validation_images.npy')
Y_valid = np.load('mnist_validation_labels.npy')

acc = compute_acc(X_valid, Y_valid, W, b) * 100
unreg_loss = compute_L_unreg(X_valid, Y_valid, W, b)

print("Validation Loss: %.2f" % unreg_loss)
print("Validation Accuracy: %.2f %%." % acc)

plt.plot(L)
plt.title("Loss over each iteration (mini-batch)")
plt.show()


""" Final Performance Evaluation on the Test Set """
# X_test = np.load('mnist_test_images.npy')
# Y_test = np.load('mnist_test_labels.npy')
#
# acc = compute_acc(X_test, Y_test, W) * 100
# unreg_loss = compute_L_unreg(X_test, Y_test, W)
#
# print("Final Loss: %.2f" % unreg_loss)
# print("Final Test Accuracy: %.2f %%." % acc)
# #96.5 minimum test accuracy


