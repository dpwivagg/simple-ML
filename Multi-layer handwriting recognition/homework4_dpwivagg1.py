import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

def compute_z(x_in, W, b):
    # Compute the Logit value, z = x*W.T + b
    z = np.dot(x_in, W.T) + np.tile(b, (x_in.shape[0], 1))
    return z


def compute_a_relu(z_in):
    # Compute the ReLU Activation, a = {0 for z < 0, z for z > 0}
    return np.maximum(z_in, 0)


def compute_a_softmax(z_in):
    # Compute the softmax activation, a = exp(z)/sum(exp(z)) for the output layer
    z = deepcopy(z_in)
    underflow_safe_z = z

    # b = underflow_safe_z.max()
    # underflow_safe_z = underflow_safe_z - b
    # underflow_safe_z[underflow_safe_z < -500] = -500
    y_hat = np.exp(underflow_safe_z)
    y_hat_sum = np.sum(y_hat, axis=1)
    a = y_hat.T / y_hat_sum
    a[np.isclose(a, 0.0, atol=1e-5)] = 1e-5
    return a.T


def compute_L(a_in, y_in):
    # Compute multi-class cross entropy, L = (-1/n) sum(y_k log(y_hat_k))
    # over all classes and instances
    n = y_in.shape[0] # Number of instances
    a = deepcopy(a_in)
    a = a[y_in == 1]
    cross_loss = np.log(a)
    L = -(1 / n) * cross_loss.sum()
    return L


def append_bias(a):
    return np.append(a, np.ones((a.shape[0], 1)), axis=1)


def forward(x, W, b, hidden_layers):
    W_in, W_n, W_out = W
    b_in, b_n, b_out = b

    z_in = compute_z(x, W_in, b_in.T)
    a_in = compute_a_relu(z_in)

    z_n = compute_z(a_in, W_n[:, :, 0], b_n[:, :, 0].T)
    a_n = compute_a_relu(z_n)

    z_n = np.array([z_n])
    a_n = np.array([a_n])

    for i in range(1, hidden_layers):
        z = compute_z(a_n[i - 1], W_n[:, :, i], b_n[:, :, i].T)
        a = compute_a_relu(z)

        z_n = np.append(z_n, np.array([z]), axis=0)
        a_n = np.append(a_n, np.array([a]), axis=0)

    z_out = compute_z(a_n[-1], W_out, b_out.T)
    a_out = compute_a_softmax(z_out)

    return z_in, a_in, z_n, a_n, z_out, a_out


def compute_dL_dz(a_in, y_in):
    # Compute local gradient of the multi-class cross-entropy loss function w.r.t. the outupt layer activation
    dL_da = (1 / y_in.shape[0]) * np.subtract(a_in, y_in)
    return dL_da


def compute_df_da_relu(dL_dz, W):
    dL_da = np.dot(dL_dz, W)
    return dL_da


def compute_da_relu_dz(a_in):
    # Compute local gradient of the relu activations a w.r.t. the logits z.
    da_dz = deepcopy(a_in)
    da_dz[da_dz > 0] = 1
    da_dz[da_dz < 0] = 0
    return da_dz


def compute_dz_dW(x_in):
    # Compute local gradient of the logits function z w.r.t. the weights W.
    dz_dW = x_in.T
    return dz_dW


def compute_dz_db(c):
    # Compute local gradient of the logits function z w.r.t. the biases b.
    dz_db = np.ones((c, 1))
    return dz_db


def compute_df_dz(dL_da, da_dz):
    # Given the local gradients, compute the gradient of the loss function L w.r.t. the logits z using chain rule.
    dL_dz = np.multiply(da_dz, dL_da)
    return dL_dz


def compute_df_dW(dL_dz, dz_dW, W, alpha):
    # Given the local gradients, compute the gradient of the loss function L w.r.t. the weights W using chain rule.
    dL_dW = np.dot(dz_dW, dL_dz) + alpha * W.T
    return dL_dW.T


def compute_df_db(dL_dz, dz_db):
    # Given the local gradients, compute the gradient of the loss function L w.r.t. the biases b using chain rule.
    dL_db = np.dot(dL_dz.T, dz_db)
    return dL_db


def update_W(W, dL_dW, epsilon):
    W = W - epsilon * dL_dW
    return W


def update_b(b, dL_db, epsilon):
    b = b - epsilon * dL_db
    return b


def compute_softmax_backprop(a0, a1, y, W, alpha):
    # Backpropogation for the final layer (first one to undergo update)
    # Output layer has softmax activation, so the gradient update is different
    # Equation: dL_dW = dL_dyhat * dyhat_da * da_dz * dz_dw = (1/n) x^T(y_hat - y)
    dL_dz = compute_dL_dz(a1, y)
    dz_dW = compute_dz_dW(a0)
    dz_db = compute_dz_db(y.shape[0])
    dL_dW = compute_df_dW(dL_dz, dz_dW, W, alpha)
    dL_db = compute_df_db(dL_dz, dz_db)
    # Update dL_da for the current layer to pass to the next layer
    dL_da = np.dot(dL_dz, W)

    return dL_dW, dL_db, dL_da

def compute_relu_backprop(a0, a1, W, df_da, alpha):
    # Backprop for first layer and all intermediate layers
    # Uses ReLU activation
    # Equation: dL_dW = dL_da_last * da_dz * dz_dW = gx^T
    da_dz = compute_da_relu_dz(a1)
    dz_dW = compute_dz_dW(a0)
    dz_db = compute_dz_db(a0.shape[0])
    df_dz = compute_df_dz(df_da, da_dz)
    df_dW = compute_df_dW(df_dz, dz_dW, W, alpha)
    df_db = compute_df_db(df_dz, dz_db)
    # Update dL_da for the current layer to pass to the next layer
    df_da = np.dot(df_dz, W)

    return df_dW, df_db, df_da


def backward(x, y, a, W, b, epsilon, alpha, hidden_layers):
    # Unpack tuple arguments
    a_in, a_n, a_out = a
    W_in, W_n, W_out = W
    b_in, b_n, b_out = b

    # Compute gradients of the output layer / output weights
    dL_dW, dL_db, df_da = compute_softmax_backprop(a_n[-1], a_out, y, W_out, alpha)
    W_out = update_W(W_out, dL_dW, epsilon)
    b_out = update_b(b_out, dL_db, epsilon)

    # Compute gradients of intermediate layers / weights
    for i in range(hidden_layers - 1, 0, -1):
        df_dW, df_db, df_da = compute_relu_backprop(a_n[i - 1], a_n[i], W_n[:, :, i], df_da, alpha)
        W_n[:, :, i] = update_W(W_n[:, :, i], df_dW, epsilon)
        b_n[:, :, i] = update_b(b_n[:, :, i], df_db, epsilon)

    # Compute gradients of second layer / weights
    df_dW, df_db, df_da = compute_relu_backprop(a_in, a_n[0], W_n[:, :, 0], df_da, alpha)
    W_n[:, :, 0] = update_W(W_n[:, :, 0], df_dW, epsilon)
    b_n[:, :, 0] = update_b(b_n[:, :, 0], df_db, epsilon)

    # Compute gradients of first layer / weights
    df_dW, df_db, _ = compute_relu_backprop(x, a_in, W_in, df_da, alpha)
    W_in = update_W(W_in, df_dW, epsilon)
    b_in = update_b(b_in, df_db, epsilon)

    return W_in, W_n, W_out, b_in, b_n, b_out


def train(X, Y, batch_size, learning_rate, n_epochs, L2_reg, hidden_layers, units_per_layer):
    # number of features
    p = X.shape[1]
    # number of classes
    c = Y.shape[1]
    # number of instances
    l = X.shape[0]

    # randomly initialize W and b
    W_in = np.random.normal(scale=2.2, size=(units_per_layer, p)) / 50
    b_in = np.random.normal(scale=2.2, size=(units_per_layer, 1)) / 50
    W_n = np.random.normal(scale=2.2, size=(units_per_layer, units_per_layer, hidden_layers)) / 50
    b_n = np.random.normal(scale=2.2, size=(units_per_layer, 1, hidden_layers)) / 50
    W_out = np.random.normal(scale=2.2, size=(c, units_per_layer)) / 50
    b_out = np.random.normal(scale=2.2, size=(c, 1)) / 50

    num_batches = l / batch_size
    if not num_batches.is_integer():
        # print("Not a whole number of batches. %.2f batches with batch size %d." % (num_batches, batch_size))
        pass
    num_batches = round(num_batches)

    all_L = np.array([])
    all_acc = np.array([])
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        # Create a random order for the samples
        order = np.random.permutation(l)
        for i in range(num_batches):
            elt_indicies = order[i*batch_size:(i*batch_size)+batch_size]
            x = X[elt_indicies]
            y = Y[elt_indicies]
            alpha = L2_reg / y.shape[0]
            W = W_in, W_n, W_out
            b = b_in, b_n, b_out
            z_in, a_in, z_n, a_n, z_out, a_out = forward(x, W, b, hidden_layers)
            a = a_in, a_n, a_out
            W_in, W_n, W_out, b_in, b_n, b_out = backward(x, y, a, W, b, learning_rate, alpha, hidden_layers)

        _, _, _, _, _, a_out = forward(X, (W_in, W_n, W_out), (b_in, b_n, b_out), hidden_layers)
        L = compute_L(a_out, Y)
        all_L = np.append(all_L, L)
        acc = compute_acc(a_out, Y)
        all_acc = np.append(all_acc, acc)
        pbar.set_description("Epoch %d finished. Loss: %.2f. Accuracy: %.2f. Training progress" % (epoch+1, L, acc))

    return W_in, W_n, W_out, b_in, b_n, b_out, all_acc, all_L


def compute_acc(Y_hat, Y):
    y_hat_indicies = np.argmax(Y_hat, axis=1)
    y_indicies = np.argmax(Y, axis=1)
    error = np.array([y_indicies == y_hat_indicies])
    acc = error.sum() / Y.shape[0]
    return acc


def findBestHyperparameters():
    X = np.load('mnist_train_images.npy')
    Y = np.load('mnist_train_labels.npy')
    X_valid = np.load('mnist_validation_images.npy')
    Y_valid = np.load('mnist_validation_labels.npy')

    best_hyperparams = {}
    best_hyperparams["L"] = 1.0
    best_hyperparams["acc"] = 0.50

    # IMPORTANT: Minimum number of hidden layers is 1.
    hyperparams = []
    hyperparams.append({"batch_size": 64, "learning_rate": 0.05, "n_epochs": 5, "L2_reg": 0.001,
                        "hidden_layers": 2, "units_per_layer": 50})
    hyperparams.append({"batch_size": 128, "learning_rate": 0.05, "n_epochs": 10, "L2_reg": 0.001,
                        "hidden_layers": 2, "units_per_layer": 40})
    hyperparams.append({"batch_size": 256, "learning_rate": 0.1, "n_epochs": 15, "L2_reg": 0.005,
                        "hidden_layers": 2, "units_per_layer": 30})
    hyperparams.append({"batch_size": 64, "learning_rate": 0.05, "n_epochs": 15, "L2_reg": 0.001,
                        "hidden_layers": 3, "units_per_layer": 50})
    hyperparams.append({"batch_size": 64, "learning_rate": 0.1, "n_epochs": 15, "L2_reg": 0.001,
                        "hidden_layers": 3, "units_per_layer": 40})
    hyperparams.append({"batch_size": 128, "learning_rate": 0.5, "n_epochs": 15, "L2_reg": 0.005,
                        "hidden_layers": 3, "units_per_layer": 30})
    hyperparams.append({"batch_size": 64, "learning_rate": 0.1, "n_epochs": 10, "L2_reg": 0.001,
                        "hidden_layers": 4, "units_per_layer": 40})
    hyperparams.append({"batch_size": 64, "learning_rate": 0.5, "n_epochs": 15, "L2_reg": 0.005,
                        "hidden_layers": 4, "units_per_layer": 40})
    hyperparams.append({"batch_size": 64, "learning_rate": 0.5, "n_epochs": 15, "L2_reg": 0.005,
                        "hidden_layers": 5, "units_per_layer": 40})
    hyperparams.append({"batch_size": 64, "learning_rate": 0.75, "n_epochs": 25, "L2_reg": 0.001,
                        "hidden_layers": 6, "units_per_layer": 40})

    print("Searching for best hyperparameters.")

    for configuration in hyperparams:
        print("==================================================")
        print("Testing configuration: ", configuration)

        try:
            weights = train(X, Y, **configuration)
        except:
            print("Error raised on configuration: ", configuration)
            continue

        """ Performance Evaluation with the Validation Set """
        _, _, _, _, _, a_out = forward(X_valid, weights[0:3], weights[3:6], configuration["hidden_layers"])
        L = compute_L(a_out, Y_valid)
        acc = compute_acc(a_out, Y_valid)
        print("Validation Loss: %.2f" % L)
        print("Validation Accuracy: %.3f." % acc)

        if L < best_hyperparams["L"] or acc > best_hyperparams["acc"]:
            configuration["L"] = L
            configuration["acc"] = acc
            best_hyperparams = configuration

    print("Best hyperparameters found. Configuration is: ", best_hyperparams)
    return best_hyperparams


if __name__ == "__main__":
    X = np.load('mnist_train_images.npy')
    Y = np.load('mnist_train_labels.npy')

    best_hyperparams = findBestHyperparameters()
    print("\n==================================================")
    print("Training network with best found hyperparameters")
    del best_hyperparams["L"]
    del best_hyperparams["acc"]
    weights = train(X, Y, **best_hyperparams)

    plt.plot(weights[-1])
    plt.title("Loss during training with the final parameters")
    plt.show()
    plt.plot(weights[-2])
    plt.title("Accuracy during training with the final parameters")
    plt.show()

    """ Final Performance Evaluation on the Test Set """
    X_test = np.load('mnist_test_images.npy')
    Y_test = np.load('mnist_test_labels.npy')

    _, _, _, _, _, a_out = forward(X_test, weights[0:3], weights[3:6], best_hyperparams["hidden_layers"])
    L = compute_L(a_out, Y_test)
    acc = compute_acc(a_out, Y_test)

    print("Final Parameters: ", best_hyperparams)
    print("Final Loss: %.2f" % L)
    print("Final Test Accuracy: %.3f." % acc)
