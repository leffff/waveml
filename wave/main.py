import torch
import numpy as np
from sklearn.metrics import mean_squared_log_error

def RMSE(y_true, y_pred):
    error = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    return error


class WaveRegressor():
    def __init__(self, n_opt_rounds=1000, learning_rate=0.001, loss_function=RMSE, verbose=1):
        self.n_opt_rounds = n_opt_rounds
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.verbose = verbose


    # Fuction for weight optimization
    def opt_func(self, X_segment, y_segment, w):
        y_true = y_segment
        y_pred = self.predict(X_segment)

        return self.loss_function(y_true, y_pred)

    #Training process
    def fit(self, X, y, weights=None, eval_set=None, use_best_model=False, cv=False):
        self.use_best_model = use_best_model
        is_eval_set = True if eval_set != None else False

        self.train_losses = np.array([])
        self.test_losses = np.array([])

        self.weights_history = []

        n_features = X.shape[1]
        self.weights = torch.tensor(weights) if weights != None else torch.tensor([1 / n_features for i in range(n_features)])
        self.weights.requires_grad_()
        self.optimizer = torch.optim.Adam([self.weights], self.learning_rate)

        for i in range(self.n_opt_rounds):
            # clear gradient
            self.optimizer.zero_grad()
            # get train set error
            train_loss = self.opt_func(X_segment=X, y_segment=y,
                                       w=self.weights)
            #append train loss to train loss history
            np.append(self.train_losses, train_loss.item())
            #create a train part of fit information
            train_output = f"train: {train_loss.item()}"
            # optimize weights according to the function
            train_loss.backward()

            #create a test part of fit information
            test_output = ""
            if is_eval_set:
                # get test set error
                test_loss = self.opt_func(X_segment=eval_set[0], y_segment=eval_set[1], w=self.weights)
                # append test loss to test loss history
                np.append(self.test_losses, test_loss)
                test_output = f"test: {test_loss.item()}"

            print(f"round: {i}", train_output, test_output)
            self.weights_history.append(self.weights)

            self.optimizer.step()

    # Get a tensor of weights after training
    def get_weights(self):
        if self.use_best_model:
            return self.weights_history[self.test_losses.argmax()[0]]
        return self.weights_history[self.train_losses.argmax()[0]]

    #Predict on on passed data with current weights
    def predict(self, X):
        return torch.sum(torch.tensor(X.to_numpy()) * self.weights, 1)

    def score(self):
        return

    def plot(self):
        return

# import pandas as pd
# S_train = pd.DataFrame()
# S_test = pd.DataFrame()
#
# features = [f"features {i + 1}" for i in range(7)]
#
# for i in range(len(features)):
#     S_train[features[i]] = [i + 1 for k in range(1000)]
#     S_test[features[i]] = [i + 0.5 for j in range(1000)]
#
# y = torch.tensor([4 for i in range(1000)])
#
# wr = WaveRegressor()
# wr.fit(S_train, y, weights=[0.0 for i in range(7)])