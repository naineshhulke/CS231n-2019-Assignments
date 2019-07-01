from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(num_train):
        scores = X[i].dot(W)
        temp = scores - scores[y[i]] + 1
        temp[ y[i] ] = 0
        temp[ temp < 0  ] = 0
        ds = np.zeros(temp.shape)
        ds[temp>0] = 1
        ds[ y[i] ] = -np.sum(temp)
    
        dW = dW +  np.dot(X[i].reshape((1,len(X[i]))).T,ds.reshape((1,ds.shape[0])))
    dW = dW/num_train
    dW += 2 * reg * W
        

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#     margin = 1
#     scores = np.dot(X,W)
#     ds = np.ones(scores.shape)
#     svm_scor = scores
#     num_train = X.shape[0]
    
#     for i in range(0,X.shape[0]):
#         act_class = svm_scor[i][y[i]]
#         svm_scor[i] = svm_scor[i] - act_class + margin
#         svm_scor[i][y[i]] = 0
#         ds[i][y[i]] = -1*( X.shape[1] - 1 )
#     svm_scor = np.clip( svm_scor , 0 , None )
#     loss = np.sum( svm_scor )/num_train + reg*np.sum(W*W)

    scores = np.dot(X,W)
    y_score = scores[ range(y.shape[0]) , y[range(y.shape[0])]  ]
    temp = scores - y_score.reshape((y_score.shape[0],1)) + 1
    temp[ range(y.shape[0]) , y[range(y.shape[0])] ] = 0
    temp[ temp < 0  ] = 0
    loss = np.sum(temp)
    loss /= X.shape[0]

    loss += reg * np.sum(W * W)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW = dW
    
    ds = np.zeros(temp.shape)
    ds[temp>0] = 1
    ds[ range(y.shape[0]) , y[range(y.shape[0])] ] = -np.sum(temp,axis =1)
    
    dW = np.dot(X.T,ds)
    dW /= X.shape[0]
    dW += 2 * reg * W
    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
