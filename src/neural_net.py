import numpy as np
from math import sqrt

def forward(input):
    a = input
    pre_activations = []
    activations = [a]
    count = 0
    for w, b in zip(weights, biases):
        z = np.dot(w, a) + b

        a = relu(z)
        pre_activations.append(z)
        activations.append(a)
    return a, pre_activations, activations


def activation(z, derivative=False):

    if derivative:
        return activation(z) * (1 - activation(z))
    else:
        return 1 / (1 + np.exp(-z))

def relu(Z, derivative = False):
    if derivative:
        return (Z > 0).astype(int)
    else:
        return np.maximum(0,Z)
    

def cost_function(y_true, y_pred):
   
    cost = sqrt(np.mean((y_true - y_pred) ** 2))
    return cost

def cost_function_prime(y_true, y_pred):
    #dL/dZ
    cost_prime = y_pred - y_true
    return cost_prime


def compute_deltas(pre_activations, y_true, y_pred):
    delta_L = cost_function_prime(y_true, y_pred) * relu(pre_activations[-1], derivative=True)
#     print(size) [2,2,1]
    deltas = [0] * (len(size) - 1)
    deltas[-1] = delta_L
    for l in range(len(deltas) - 2, -1, -1):
        delta = np.dot(weights[l + 1].transpose(), deltas[l + 1]) * activation(pre_activations[l], derivative=True) 
        deltas[l] = delta
    return deltas


def backpropagate( deltas, pre_activations, activations):
    dW = []
    db = []
    deltas = [0] + deltas
    for l in range(1, len(size)):
        dW_l = np.dot(deltas[l], activations[l-1].transpose()) 
        db_l = deltas[l]
        dW.append(dW_l)
        db.append(np.expand_dims(db_l.mean(axis=1), 1))
    return dW, db

def predict( a):
    for w, b in zip(weights, biases):
        z = np.dot(w, a) + b
        a = relu(z)
    predictions = (a > 0.5).astype(int)
    return predictions


if __name__=="__main__":
    size = 1000
    train_x = np.random.rand(size,2) * 2 - 1
    r = np.dot(train_x, np.array([[1,1],[1,-1]])) + np.random.randn(size, 2) * 0.05
    train_y = np.logical_xor(r[:, 0]>0, r[:, 1]>0).astype(int)

    X = train_x.T
    y = np.expand_dims(train_y, 1).T


    np.random.seed(42)
    size = [2, 2, 1]
    weights = [np.random.randn(size[i], size[i-1]) * np.sqrt(1 / size[i-1]) for i in range(1, len(size))]
    biases = [np.random.rand(n, 1) for n in size[1:]]

    print_every=10
    batch_size=16
    epochs=1000
    learning_rate=0.1

    x_train = X
    y_train = y

    epoch_iterator = range(epochs)

    for e in epoch_iterator:
        if x_train.shape[1] % batch_size == 0:
            n_batches = int(x_train.shape[1] / batch_size)
        else:
            n_batches = int(x_train.shape[1] / batch_size ) - 1
        batches_x = [x_train[:, batch_size*i:batch_size*(i+1)] for i in range(0, n_batches)]
        batches_y = [y_train[:, batch_size*i:batch_size*(i+1)] for i in range(0, n_batches)]

        train_losses = []
        train_accuracies = []
        
        test_losses = []
        test_accuracies = []
        
        dw_per_epoch = [np.zeros(w.shape) for w in weights]
        db_per_epoch = [np.zeros(b.shape) for b in biases] 
    #     print(dw_per_epoch)
    #     print(db_per_epoch)
        for batch_x, batch_y in zip(batches_x, batches_y):
            batch_y_pred, pre_activations, activations = forward(batch_x)
            deltas = compute_deltas(pre_activations, batch_y, batch_y_pred)
            dW, db = backpropagate(deltas, pre_activations, activations)
            for i, (dw_i, db_i) in enumerate(zip(dW, db)):
                dw_per_epoch[i] += dw_i / batch_size
                db_per_epoch[i] += db_i / batch_size
            batch_y_train_pred = predict(batch_x)
            train_loss = cost_function(batch_y, batch_y_train_pred)
            train_losses.append(train_loss)
            train_accuracy = (batch_y == batch_y_train_pred)
            train_accuracies.append(train_accuracy)

        # weight update
        for i, (dw_epoch, db_epoch) in enumerate(zip(dw_per_epoch, db_per_epoch)):
            weights[i] = weights[i] - learning_rate * dw_epoch
            biases[i] = biases[i] - learning_rate * db_epoch
        
        
        print('Epoch {} / {} | train loss: {} | train accuracy: {} '.format(
                    e, epochs, np.round(np.mean(train_losses), 3), np.round(np.mean(train_accuracies), 3)))


    size = 200
    test_x = np.random.rand(size,2) * 2 - 1
    r = np.dot(test_x, np.array([[1,1],[1,-1]])) + np.random.randn(size, 2) * 0.05
    test_y = np.logical_xor(r[:, 0]>0, r[:, 1]>0).astype(int)
    pred_y = np.zeros((size,))

    y_test_pred = predict(test_x.T)
    test_loss = cost_function(test_y, y_test_pred)
    test_losses.append(test_loss)
    test_accuracy = (test_y == y_test_pred)

    print(f'The accuracy of the model is {(1-((y_test_pred>0.5) ^ test_y)).mean()}')

