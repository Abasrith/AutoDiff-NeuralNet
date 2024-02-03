import numpy as np
import argparse
from typing import Callable, List, Tuple

# This takes care of command line argument parsing for you!
# To access a specific argument, simply access args.<argument name>.
parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')


def args2data(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
str, str, str, int, int, int, float]:
    """
    Parse command line arguments, create train/test data and labels.
    :return:
    X_tr: train data *without label column and without bias folded in
        (numpy array)
    y_tr: train label (numpy array)
    X_te: test data *without label column and without bias folded in*
        (numpy array)
    y_te: test label (numpy array)
    out_tr: file for predicted output for train data (file)
    out_te: file for predicted output for test data (file)
    out_metrics: file for output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """
    # Get data from arguments
    out_tr = args.train_out
    out_te = args.validation_out
    out_metrics = args.metrics_out
    n_epochs = args.num_epoch
    n_hid = args.hidden_units
    init_flag = args.init_flag
    lr = args.learning_rate

    X_tr = np.loadtxt(args.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr = X_tr[:, 1:]  # cut off label column

    X_te = np.loadtxt(args.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te = X_te[:, 1:]  # cut off label column

    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)


def shuffle(X, y, epoch):
    """
    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]


def zero_init(shape):
    """
    ZERO Initialization: All weights are initialized to 0.

    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    return np.zeros(shape=shape)


def random_init(shape):
    """

    RANDOM Initialization: The weights are initialized randomly from a uniform
        distribution from -0.1 to 0.1.

    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    M, D = shape
    np.random.seed(M * D)  
 
    return np.random.uniform(-0.1, 0.1, size=(M, D))


class SoftMaxCrossEntropy:

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Implement softmax function.
        :param z: input logits of shape (num_classes,)
        :return: softmax output of shape (num_classes,)
        """
        e_z = np.exp(z)
        softMax = e_z/e_z.sum(axis=0)
        return softMax

    def _cross_entropy(self, y: int, y_hat: np.ndarray) -> float:
        """
        Compute cross entropy loss.
        :param y: integer class label
        :param y_hat: prediction with shape (num_classes,)
        :return: cross entropy loss
        """
        crossEntropyLoss = -np.log(y_hat[y])
        return crossEntropyLoss

    def forward(self, z: np.ndarray, y: int) -> Tuple[np.ndarray, float]:
        """
        Compute softmax and cross entropy loss.
        :param z: input logits of shape (num_classes,)
        :param y: integer class label
        :return:
            y: predictions from softmax as an np.ndarray
            loss: cross entropy loss
        """
        y_pred = self._softmax(z)
        lossVal = self._cross_entropy(y, y_pred)
        return (y_pred, lossVal)

    def backward(self, y: int, y_hat: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. ** softmax input **.

        :param y: integer class label
        :param y_hat: predicted softmax probability with shape (num_classes,)
        :return: gradient with shape (num_classes,)
        """
        yOneHot = np.zeros(y_hat.shape[0])
        yOneHot[y] = 1
        grad_b = np.subtract(y_hat, yOneHot)
        grad_b = np.expand_dims(grad_b, axis=1)
        return grad_b


class Sigmoid:
    def __init__(self):
        """
        Initialize state for sigmoid activation layer
        """
        self.sigmoidOutVal = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Take sigmoid of input x.
        :param x: Input to activation function (i.e. output of the previous 
                  linear layer), with shape (output_size,)
        :return: Output of sigmoid activation function with shape
            (output_size,)
        """
        e = np.exp(x)
        sigmoid_x  = e / (1 + e)
        self.sigmoidOutVal = sigmoid_x
        return sigmoid_x

    def backward(self, dz: np.ndarray) -> np.ndarray:
        """
        :param dz: partial derivative of loss with respect to output of
            sigmoid activation
        :return: partial derivative of loss with respect to input of
            sigmoid activation
        """
        sigmoidOutVal_partialSub = np.subtract(1, self.sigmoidOutVal)
        sigmoidOutVal_partialProd = np.multiply(self.sigmoidOutVal, sigmoidOutVal_partialSub)
        sigmoidOutVal_partialProd = np.expand_dims(sigmoidOutVal_partialProd, axis=1)
        grad_sigmoid = np.multiply(dz, sigmoidOutVal_partialProd)
        return grad_sigmoid


# This refers to a function type that takes in a tuple of 2 integers (row, col)
# and returns a numpy array (which should have the specified dimensions).
INIT_FN_TYPE = Callable[[Tuple[int, int]], np.ndarray]

class Linear:
    def __init__(self, input_size: int, output_size: int,
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        """
        :param input_size: number of units in the input of the layer 
                           *not including* the folded bias
        :param output_size: number of units in the output of the layer
        :param weight_init_fn: function that creates and initializes weight 
                               matrices for layer. This function takes in a 
                               tuple (row, col) and returns a matrix with
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        """
        # Initialize learning rate for SGD
        self.lr = learning_rate

        self.weights = weight_init_fn((output_size, input_size+1))

        # set the bias terms to zero
        # print(f"bias column = {np.zeros(self.weights.shape[0]).shape}")
        # print(f"orin weights column = {self.weights.shape}")
        #self.weights = np.column_stack((np.zeros(self.weights.shape[0]), self.weights))
        self.weights[:,0] = 0

        # Initialize matrix to store gradient with respect to weights
        self.dw = np.zeros((output_size, input_size+1))


        # Initialize any additional values you may need to store for the
        #  backward pass here
        self.LinearInput = None


    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: Input to linear layer with shape (input_size,)
                  where input_size *does not include* the folded bias.

        :return: output z of linear layer with shape (output_size,)

        """
        xInclBias = np.insert(x, 0, 1)
        self.LinearInput = np.expand_dims(xInclBias, axis=1)
        linearForwardPass = np.dot(self.weights, xInclBias)
        return linearForwardPass

    def backward(self, dz: np.ndarray) -> np.ndarray:
        """
        :param dz: partial derivative of loss with respect to output z
            of linear
        :return: dx, partial derivative of loss with respect to input x
            of linear
        
        """
        self.dw = np.dot(dz, self.LinearInput.T)
        betaTranspose = self.weights[:, 1:].T
        linearGradIn = np.dot(betaTranspose, dz)
        return (linearGradIn)

    def step(self) -> None:
        """
        Apply SGD update to weights using self.dw, which should have been 
        set in NN.backward().
        """
        partialProd_lr_dw = np.multiply(self.lr, self.dw)
        self.weights = np.subtract(self.weights, partialProd_lr_dw)


class NN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        """
        Initalize neural network (NN) class. Note that this class is composed
        of the layer objects (Linear, Sigmoid) defined above.

        :param input_size: number of units in input to network
        :param hidden_size: number of units in the hidden layer of the network
        :param output_size: number of units in output of the network - this
                            should be equal to the number of classes
        :param weight_init_fn: function that creates and initializes weight 
                               matrices for layer. This function takes in a 
                               tuple (row, col) and returns a matrix with 
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        """
        self.weight_init_fn = weight_init_fn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_layer_1 = Linear(input_size, hidden_size, weight_init_fn, learning_rate)
        self.sigmoid_layer = Sigmoid()
        self.linear_layer_2 = Linear(hidden_size, output_size, weight_init_fn, learning_rate)
        self.softMax_layer = SoftMaxCrossEntropy()

    def forward(self, x: np.ndarray, y: int) -> Tuple[np.ndarray, float]:
        """
        :param x: input data point *without the bias folded in*
        :param y: prediction with shape (num_classes,)
        :return:
            y_hat: output prediction with shape (num_classes,). This should be
                a valid probability distribution over the classes.
            loss: the cross_entropy loss for a given example
        """
        # call forward pass for each layer
        a = self.linear_layer_1.forward(x)
        z = self.sigmoid_layer.forward(a)
        b = self.linear_layer_2.forward(z)
        y_pred, lossVal = self.softMax_layer.forward(b, y)
        return (y_pred, lossVal)

    def backward(self, y: int, y_hat: np.ndarray) -> None:
        """
        :param y: label (a number or an array containing a single element)
        :param y_hat: prediction with shape (num_classes,)
        """
        # call backward pass for each layer
        gb = self.softMax_layer.backward(y, y_hat)
        gz = self.linear_layer_2.backward(gb)
        ga = self.sigmoid_layer.backward(gz)
        gx = self.linear_layer_1.backward(ga)

    def step(self):
        """
        Apply SGD update to weights.
        """
        # call step for each relevant layer
        self.linear_layer_1.step()
        self.linear_layer_2.step()

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute nn's average (cross entropy) loss over the dataset (X, y)
        :param X: Input dataset of shape (num_points, input_size)
        :param y: Input labels of shape (num_points,)
        :return: Mean cross entropy loss
        """
        crossEntropyLossAccum = 0 
        for i in range(X.shape[0]):
            _, loss = self.forward(X[i], y[i])
            crossEntropyLossAccum += loss
        meancrossEntropyLoss = crossEntropyLossAccum/X.shape[0]
        return meancrossEntropyLoss

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              n_epochs: int) -> Tuple[List[float], List[float]]:
        """
        Train the network using SGD for some epochs.
        :param X_tr: train data
        :param y_tr: train label
        :param X_test: train data
        :param y_test: train label
        :param n_epochs: number of epochs to train for
        :return:
            train_losses: Training losses *after* each training epoch
            test_losses: Test losses *after* each training epoch
        """
        # train network
        meanTrainCrossLosses = []
        meanTestCrossLosses = []
        for e in range(n_epochs):
            x, y = shuffle(X_tr, y_tr, e)
            for i in range(x.shape[0]):
                y_hat,_ = self.forward(x[i], y[i])
                self.backward(y[i], y_hat)
                self.step()
            meanTrainCrossLosses.append(self.compute_loss(x, y))
            meanTestCrossLosses.append(self.compute_loss(X_test, y_test))
        return(meanTrainCrossLosses, meanTestCrossLosses)

    def test(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the label and error rate.
        :param X: input data
        :param y: label
        :return:
            labels: predicted labels
            error_rate: prediction error rate
        """
        # make predictions and compute error
        y_labels = np.array([])
        errorCount = 0
        for i in range(X.shape[0]):
            y_hat,_ = self.forward(X[i], y[i])
            y_labels =  np.append(y_labels, np.argmax(y_hat))
            if np.argmax(y_hat) != y[i]:
               errorCount += 1
        testError = errorCount/X.shape[0]
        return (y_labels, testError)


if __name__ == "__main__":
    args = parser.parse_args()

    # Define our labels
    labels = ["a", "e", "g", "i", "l", "n", "o", "r", "t", "u"]

    (X_tr, y_tr, X_test, y_test, out_tr, out_te, out_metrics,
     n_epochs, n_hid, init_flag, lr) = args2data(args)

    nn = NN(
        input_size=X_tr.shape[-1],
        hidden_size=n_hid,
        output_size=len(labels),
        weight_init_fn=zero_init if init_flag == 2 else random_init,
        learning_rate=lr
    )

    # train model
    train_losses, test_losses = nn.train(X_tr, y_tr, X_test, y_test, n_epochs)

    # test model and get predicted labels and errors 
    train_labels, train_error_rate = nn.test(X_tr, y_tr)
    test_labels, test_error_rate = nn.test(X_test, y_test)

    train_labels = train_labels.astype(int)
    test_labels = test_labels.astype(int)

    with open(out_tr, "w") as f:
        for label in train_labels:
            f.write(str(label) + "\n")
    with open(out_te, "w") as f:
        for label in test_labels:
            f.write(str(label) + "\n")
    with open(out_metrics, "w") as f:
        for i in range(len(train_losses)):
            cur_epoch = i + 1
            cur_tr_loss = train_losses[i]
            cur_te_loss = test_losses[i]
            f.write("epoch={} crossentropy(train): {}\n".format(
                cur_epoch, cur_tr_loss))
            f.write("epoch={} crossentropy(validation): {}\n".format(
                cur_epoch, cur_te_loss))
        f.write("error(train): {}\n".format(train_error_rate))
        f.write("error(validation): {}\n".format(test_error_rate))
