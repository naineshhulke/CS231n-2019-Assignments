from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    n = X.shape[0]
    c = W.shape[1]
    
    for i in range(0,n):
        score = np.exp ( np.dot( X[i],W ) )
        cors = score[y[i]]
        total = np.sum(score)
        fraction = cors/total
        loss = loss -np.log(fraction)
        ds = score/total
        ds[y[i]] = - (total - cors) / total
        dW = dW +  np.dot(X[i].reshape((1,X.shape[1])).T,ds.reshape((1,ds.shape[0])))
        
    loss/= n
    loss += reg*np.sum(W*W)
    dW/= n
    dW += 2*reg*W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    n = X.shape[0]
    scores = X.dot(W)
    scores -= scores.max()
    scores = np.exp(scores)
    scores_sums = np.sum(scores, axis=1)
    cors = scores[range(n), y]
    loss = cors / scores_sums
    loss = -np.sum(np.log(loss))/n + reg * np.sum(W * W)

    s = np.divide(scores, scores_sums.reshape(n, 1))
    s[range(n), y] = - (scores_sums - cors) / scores_sums
    dW = X.T.dot(s)
    dW /= n
    dW += 2 * reg * W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
