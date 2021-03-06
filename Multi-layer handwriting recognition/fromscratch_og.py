import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
from tqdm import tqdm

# -----------------------------------------------------------------
# Forward Pass
# -----------------------------------------------------------------

# -----------------------------------------------------------------
def compute_z(x, W):
    '''
        Compute the linear logit values of a data instance. z =  W x + b
        Input:
            x: the feature vector of a data instance, a float numpy matrix of shape p by 1. Here p is the number of features/dimensions.
            W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of shape c by 1.
        Output:
            z: the linear logits, a float numpy vector of shape c by 1.
        Hint: you could solve this problem using 1 line of code.
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    z = np.dot(x, W.T)

    #########################################
    return z


# -----------------------------------------------------------------
def compute_a(z):
    '''
        Compute the softmax activations.
        Input:
            z: the logit values of softmax regression, a float numpy vector of shape c by 1. Here c is the number of classes
        Output:
            a: the softmax activations, a float numpy vector of shape c by 1.
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    # Normalize to avoid overflow--we can handle underflow by just setting to a very small number

    z[z < 0] = 0
    return z

    #########################################


def compute_softmax(z):
    y_hat = np.exp(z)
    y_hat_sum = np.sum(y_hat, axis=1)
    a = y_hat.T / y_hat_sum
    return a.T


# -----------------------------------------------------------------
def compute_L(a, y):
    '''
        Compute multi-class cross entropy, which is the loss function of softmax regression.
        Input:
            a: the activations of a training instance, a float numpy vector of shape c by 1. Here c is the number of classes.
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
        Output:
            L: the loss value of softmax regression, a float scalar.
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    n = y.shape[0]
    a[a == 0] = 1e-100
    a = a[y == 1]
    cross_loss = np.log(a)

    L = -(1 / (2 * n)) * cross_loss.sum()

    return L



# -----------------------------------------------------------------
def forward(x, y, W):
    '''
       Forward pass: given an instance in the training data, compute the logits z, activations a and multi-class cross entropy L on the instance.
        Input:
            x: the feature vector of a training instance, a float numpy vector of shape p by 1. Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0 or 1.
            W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of shape c by 1.
        Output:
            z: the logit values of softmax regression, a float numpy vector of shape c by 1. Here c is the number of classes
            a: the activations of a training instance, a float numpy vector of shape c by 1. Here c is the number of classes.
            L: the loss value of softmax regression, a float scalar.
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    z = compute_z(x, W)
    a = compute_a(z)
    L = compute_L(a, y)

    #########################################
    return z, a, L


# -----------------------------------------------------------------
# Compute Local Gradients
# -----------------------------------------------------------------


# -----------------------------------------------------------------
def compute_dL_da(a, y):
    '''
        Compute local gradient of the multi-class cross-entropy loss function w.r.t. the activations.
        Input:
            a: the activations of a training instance, a float numpy vector of shape c by 1. Here c is the number of classes.
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
        Output:
            dL_da: the local gradients of the loss function w.r.t. the activations, a float numpy vector of shape c by 1.
                   The i-th element dL_da[i] represents the partial gradient of the loss function w.r.t. the i-th activation a[i]:  d_L / d_a[i].
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    dL_da = (1 / y.shape[0]) * np.subtract(a, y)

    #########################################
    return dL_da


# -----------------------------------------------------------------
def compute_da_dz(a):
    '''
        Compute local gradient of the softmax activations a w.r.t. the logits z.
        Input:
            a: the activation values of softmax function, a numpy float vector of shape c by 1. Here c is the number of classes.
        Output:
            da_dz: the local gradient of the activations a w.r.t. the logits z, a float numpy matrix of shape (c by c).
                   The (i,j)-th element of da_dz represents the partial gradient ( d_a[i]  / d_z[j] )
        Hint: you could solve this problem using 4 or 5 lines of code.
        (3 points)
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    a[a > 0] = 1
    a[a < 0] = 0
    da_dz = a

    #########################################
    return da_dz


# -----------------------------------------------------------------
def compute_dz_dW(x, c=0):
    '''
        Compute local gradient of the logits function z w.r.t. the weights W.
        Input:
            x: the feature vector of a data instance, a float numpy vector of shape p by 1. Here p is the number of features/dimensions.
            c: the number of classes, an integer.
        Output:
            dz_dW: the partial gradient of logits z w.r.t. the weight matrix, a numpy float matrix of shape (c by p).
                   The (i,j)-th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[i,j]:   d_z[i] / d_W[i,j]
        Hint: the partial gradients only depend on the input x and the number of classes
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    dz_dW = x.T

    #########################################
    return dz_dW


# -----------------------------------------------------------------
def compute_dz_db(c):
    '''
        Compute local gradient of the logits function z w.r.t. the biases b.
        Input:
            c: the number of classes, an integer.
        Output:
            dz_db: the partial gradient of the logits z w.r.t. the biases b, a float vector of shape c by 1.
                   Each element dz_db[i] represents the partial gradient of the i-th logit z[i] w.r.t. the i-th bias b[i]:  d_z[i] / d_b[i]
        Hint: you could solve this problem using 1 line of code.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    dz_db = np.ones((c, 1)) # If error on this line, add np.asmatrix around np.ones

    #########################################
    return dz_db


# -----------------------------------------------------------------
# Back Propagation
# -----------------------------------------------------------------

# -----------------------------------------------------------------
def backward(x, y, a):
    '''
       Back Propagation: given an instance in the training data, compute the local gradients of the logits z, activations a, weights W and biases b on the instance.
        Input:
            x: the feature vector of a training instance, a float numpy vector of shape p by 1. Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
            a: the activations of a training instance, a float numpy vector of shape c by 1. Here c is the number of classes.
        Output:
            dL_da: the local gradients of the loss function w.r.t. the activations, a float numpy vector of shape c by 1.
                   The i-th element dL_da[i] represents the partial gradient of the loss function L w.r.t. the i-th activation a[i]:  d_L / d_a[i].
            da_dz: the local gradient of the activation w.r.t. the logits z, a float numpy matrix of shape (c by c).
                   The (i,j)-th element of da_dz represents the partial gradient ( d_a[i]  / d_z[j] )
            dz_dW: the partial gradient of logits z w.r.t. the weight matrix W, a numpy float matrix of shape (c by p).
                   The i,j -th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[i,j]:   d_z[i] / d_W[i,j]
            dz_db: the partial gradient of the logits z w.r.t. the biases b, a float vector of shape c by 1.
                   Each element dz_db[i] represents the partial gradient of the i-th logit z[i] w.r.t. the i-th bias:  d_z[i] / d_b[i]
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    dL_da = compute_dL_da(a, y)
    da_dz = compute_da_dz(a)
    dz_dW = compute_dz_dW(x)
    dz_db = compute_dz_db(y.shape[0])

    #########################################
    return dL_da, da_dz, dz_dW, dz_db


# -----------------------------------------------------------------
def compute_dL_dz(dL_da, da_dz):
    '''
       Given the local gradients, compute the gradient of the loss function L w.r.t. the logits z using chain rule.
        Input:
            dL_da: the local gradients of the loss function w.r.t. the activations, a float numpy vector of shape c by 1.
                   The i-th element dL_da[i] represents the partial gradient of the loss function L w.r.t. the i-th activation a[i]:  d_L / d_a[i].
            da_dz: the local gradient of the activation w.r.t. the logits z, a float numpy matrix of shape (c by c).
                   The (i,j)-th element of da_dz represents the partial gradient ( d_a[i]  / d_z[j] )
        Output:
            dL_dz: the gradient of the loss function L w.r.t. the logits z, a numpy float vector of shape c by 1.
                   The i-th element dL_dz[i] represents the partial gradient of the loss function L w.r.t. the i-th logit z[i]:  d_L / d_z[i].
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    dL_dz = np.multiply(da_dz, dL_da)

    #########################################
    return dL_dz


# -----------------------------------------------------------------
def compute_dL_dW(dL_dz, dz_dW):
    '''
       Given the local gradients, compute the gradient of the loss function L w.r.t. the weights W using chain rule.
        Input:
            dL_dz: the gradient of the loss function L w.r.t. the logits z, a numpy float vector of shape c by 1.
                   The i-th element dL_dz[i] represents the partial gradient of the loss function L w.r.t. the i-th logit z[i]:  d_L / d_z[i].
            dz_dW: the partial gradient of logits z w.r.t. the weight matrix W, a numpy float matrix of shape (c by p).
                   The i,j -th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[i,j]:   d_z[i] / d_W[i,j]
        Output:
            dL_dW: the global gradient of the loss function w.r.t. the weight matrix, a numpy float matrix of shape (c by p).
                   Here c is the number of classes.
                   The i,j-th element dL_dW[i,j] represents the partial gradient of the loss function L w.r.t. the i,j-th weight W[i,j]:  d_L / d_W[i,j]
        Hint: you could solve this problem using 2 lines of code
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    dL_dW = np.dot(dz_dW, dL_dz)

    #########################################
    return dL_dW.T


# -----------------------------------------------------------------
def compute_dL_db(dL_dz, dz_db):
    '''
       Given the local gradients, compute the gradient of the loss function L w.r.t. the biases b using chain rule.
        Input:
            dL_dz: the gradient of the loss function L w.r.t. the logits z, a numpy float vector of shape c by 1.
                   The i-th element dL_dz[i] represents the partial gradient of the loss function L w.r.t. the i-th logit z[i]:  d_L / d_z[i].
            dz_db: the local gradient of the logits z w.r.t. the biases b, a float numpy vector of shape c by 1.
                   The i-th element dz_db[i] represents the partial gradient ( d_z[i]  / d_b[i] )
        Output:
            dL_db: the global gradient of the loss function L w.r.t. the biases b, a float numpy vector of shape c by 1.
                   The i-th element dL_db[i] represents the partial gradient of the loss function w.r.t. the i-th bias:  d_L / d_b[i]
        Hint: you could solve this problem using 1 line of code in the block.
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    dL_db = np.dot(dL_dz.T, dz_db)

    #########################################
    return dL_db


# -----------------------------------------------------------------
# gradient descent
# -----------------------------------------------------------------

# --------------------------
def update_W(W, dL_dW, dL_db, epsilon=0.001):
    '''
       Update the weights W using gradient descent.
        Input:
            W: the current weight matrix, a float numpy matrix of shape (c by p). Here c is the number of classes.
            alpha: the step-size parameter of gradient descent, a float scalar.
            dL_dW: the global gradient of the loss function w.r.t. the weight matrix, a numpy float matrix of shape (c by p).
                   The i,j-th element dL_dW[i,j] represents the partial gradient of the loss function L w.r.t. the i,j-th weight W[i,j]:  d_L / d_W[i,j]
        Output:
            W: the updated weight matrix, a numpy float matrix of shape (c by p).
        Hint: you could solve this problem using 1 line of code
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    W[:,0:-1] = W[:,0:-1] - epsilon * dL_dW[:,0:-1]
    W[:,-1] = W[:,-1] - epsilon * dL_db[:,-1]

    #########################################
    return W


# --------------------------
def update_b(b, dL_db, alpha=0.001):
    '''
       Update the biases b using gradient descent.
        Input:
            b: the current bias values, a float numpy vector of shape c by 1.
            dL_db: the global gradient of the loss function L w.r.t. the biases b, a float numpy vector of shape c by 1.
                   The i-th element dL_db[i] represents the partial gradient of the loss function w.r.t. the i-th bias:  d_L / d_b[i]
            alpha: the step-size parameter of gradient descent, a float scalar.
        Output:
            b: the updated of bias vector, a float numpy vector of shape c by 1.
        Hint: you could solve this problem using 1 lines of code
    '''

    #########################################
    ## INSERT YOUR CODE HERE

    b = b - alpha * dL_db

    #########################################
    return b


# --------------------------
# train
def train(X, Y, batch_size=75, epsilon=0.005, n_epochs=2, alpha=.005):
    '''
       Given a training dataset, train the softmax regression model by iteratively updating the weights W and biases b using the gradients computed over each data instance.
        Input:
            X: the feature matrix of training instances, a float numpy matrix of shape (n by p). Here n is the number of data instance in the training set, p is the number of features/dimensions.
            Y: the labels of training instance, a numpy integer numpy array of length n. The values can be 0 or 1.
            alpha: the step-size parameter of gradient ascent, a float scalar.
            n_epoch: the number of passes to go through the training set, an integer scalar.
        Output:
            W: the weight matrix trained on the training set, a numpy float matrix of shape (c by p).
            b: the bias, a float numpy vector of shape c by 1.
    '''
    # number of features
    p = X.shape[1]
    # number of classes
    c = Y.shape[1]
    # number of instances
    l = X.shape[0]

    # randomly initialize W and b
    # W = np.asmatrix(np.random.rand(c, p))
    W = np.zeros((c, p))
    b = W[:, -1]
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
            y = Y[elt_indicies]

            # Forward pass: compute the logits, softmax and cross_entropy
            z, a, _ = forward(x, y, W)

            # Back Propagation: compute local gradients of cross_entropy, softmax and logits
            dL_da, da_dz, dz_dW, dz_db = backward(x, y, a)

            # compute the global gradients using chain rule
            dL_dz = compute_dL_dz(dL_da, da_dz)
            dL_dW = compute_dL_dW(dL_dz, dz_dW)
            dL_db = compute_dL_db(dL_dz, dz_db)

            # update the paramters using gradient descent
            W = update_W(W, dL_dW, dL_db, epsilon)
            # b = update_b(b, dL_db, epsilon)

            _, _, L = forward(X, Y, W)
            L_vec = np.append(L_vec, L)
            acc = compute_acc(X, Y, W)
            pbar.set_description("Loss: %.2f. Accuracy: %.2f. Epoch progress" % (L, acc))

        acc = compute_acc(X, Y, W)
        print("\nEpoch %d completed. Epoch loss: %.2f. Epoch accuracy: %.2f\n" % (epoch+1, L, acc))

    return W, L_vec


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

W, L = train(X, Y, batch_size=75, epsilon=0.001, n_epochs=1, alpha=.005)

plt.plot(L)
plt.title("Loss over each iteration (mini-batch)")
plt.show()

""" Performance Evaluation with the Validation Set """
X_valid = np.load('mnist_validation_images.npy')
X_valid = np.append(X_valid, np.ones((X_valid.shape[0], 1)), axis=1)
Y_valid = np.load('mnist_validation_labels.npy')

acc = compute_acc(X_valid, Y_valid, W) * 100
valid_loss = compute_L(compute_a(compute_z(X_valid, W)), Y)

print("Validation Loss: %.2f" % valid_loss)
print("Validation Accuracy: %.2f %%." % acc)
