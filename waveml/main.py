import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from waveml.metrics import RMSE, MSE, MAE

def to_tensor(X: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list]) -> torch.tensor:
    dtype = type(X)

    if dtype == pd.DataFrame:
        return torch.tensor(X.to_numpy())

    elif dtype == pd.Series:
        return torch.tensor(X.values)

    elif dtype == np.ndarray:
        return torch.tensor(X)

    elif dtype == list:
        return torch.tensor(X)

    return X


class Wave:
    def __init__(self, n_opt_rounds=1000, learning_rate=0.01, loss_function=MSE, verbose=1):
        self.n_opt_rounds = int(n_opt_rounds)
        self.learning_rate = float(learning_rate)
        self.loss_function = loss_function
        self.verbose = int(verbose)
        self.fitted = False

        if self.n_opt_rounds < 1:
            raise ValueError(f"n_opt_rounds should belong to an [1;inf) interval, passed {self.n_opt_rounds}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning rate should belong to a (0;inf) interval, passed {self.learning_rate}")
        if self.verbose < 0:
            raise ValueError(f"learning rate should belong to a [0;inf) interval, passed {self.verbose}")


class WaveRegressor(Wave):
    def __init__(self, n_opt_rounds=1000, learning_rate=0.01, loss_function=MSE, verbose=1):
        super().__init__(n_opt_rounds, learning_rate, loss_function, verbose)
    # Training process
    def fit(self, X, y, weights=None, eval_set=None, use_best_model=False) -> None:
        X_train_tensor, y_train_tensor, self.use_best_model = to_tensor(X), to_tensor(y), use_best_model
        self.train_losses, self.test_losses, self.weights_history = [], [], []

        if type(self.use_best_model) != bool:
            raise ValueError(f"use_best_model parameter should be bool, passed {self.use_best_model}")

        self.is_eval_set = True if eval_set != None else False
        if self.is_eval_set:
            X_test_tensor = to_tensor(eval_set[0])
            y_test_tensor = to_tensor(eval_set[1])

        n_features = X_train_tensor.shape[1]
        self.weights = to_tensor(weights) if weights != None else torch.tensor(
            [1 / n_features for i in range(n_features)]
        )

        self.weights.requires_grad_()
        self.optimizer = torch.optim.Adam([self.weights], self.learning_rate)

        for i in range(self.n_opt_rounds):
            # clear gradient
            self.optimizer.zero_grad()
            # get train set error
            train_loss = self.__opt_func(X_segment=X_train_tensor, y_segment=y_train_tensor)
            # append train loss to train loss history
            self.train_losses.append(train_loss.item())
            # create a train part of fit information
            train_output = f"train: {train_loss.item()}"
            # optimize weights according to the function
            train_loss.backward()

            # create a test part of fit information
            test_output = ""
            if self.is_eval_set:
                # get test set error
                test_loss = self.__opt_func(X_segment=X_test_tensor, y_segment=y_test_tensor)
                # append test loss to test loss history
                self.test_losses.append(test_loss.item())
                test_output = f"test: {test_loss.item()}"

            if self.verbose != 0:
                print(f"round: {i}", train_output, test_output)
            self.weights_history.append(self.weights)

            self.optimizer.step()

        self.fitted = True

    # Get a tensor of weights after training
    def get_weights(self) -> np.ndarray:
        if not self.fitted:
            raise AttributeError("Model has not been fitted yet. Use fit() method first.")

        if self.use_best_model:
            return self.weights_history[self.test_losses.index(min(self.test_losses))].detach().numpy()
        return self.weights_history[self.train_losses.index(min(self.train_losses))].detach().numpy()

    # Predict on on passed data with current weights
    def predict(self, X) -> np.ndarray:
        if not self.fitted:
            raise AttributeError("Model has not been fitted yet. Use fit() method first.")

        X = to_tensor(X)
        sum = torch.sum(X * self.get_weights(), 1)
        return sum.detach().numpy()

    def score(self, X_train, y_test):
        if not self.fitted:
            raise AttributeError("Model has not been fitted yet. Use fit() method first.")

        X_train_tensor, y_test_tensor = to_tensor(X_train), to_tensor(y_test)
        y_pred = self.predict(X_train_tensor)
        return self.loss_function(y_test_tensor, y_pred)

    def plot(self) -> None:
        if not self.fitted:
            raise AttributeError("Model has not been fitted yet. Use fit() method first.")

        plt.plot([i for i in range(self.n_opt_rounds)], self.train_losses)
        if self.is_eval_set:
            plt.plot([i for i in range(self.n_opt_rounds)], self.test_losses)
        plt.show()
        return

    # Function for weight optimization
    def __opt_func(self, X_segment, y_segment):
        y_true = y_segment
        y_pred = self.__inner_predict(X_segment)
        return self.loss_function(y_true, y_pred)

    def __inner_predict(self, X) -> torch.tensor:
        sum = torch.sum(X * self.weights, 1)
        return sum


class WavePredictionTuner(Wave):
    def __init__(self, n_opt_rounds=1000, learning_rate=0.1, loss_function=MSE, verbose=1):
        super().__init__(n_opt_rounds, learning_rate, loss_function, verbose)

    def __opt_func(self, X_segment, y_segment, weights):
        return self.loss_function(X_segment * weights[0] + weights[1], y_segment)

    def fit(self, X, y, use_best_model=False) -> None:
        X_train_tensor, y_train_tensor, self.use_best_model = to_tensor(X), to_tensor(y), use_best_model
        self.train_losses = []
        self.n_features =  X_train_tensor.shape[1]
        self.weights = torch.tensor([])

        self.fitted = False

        for i in range(self.n_features):
            weights = torch.tensor([1.0, 0.0])
            weights.requires_grad_()
            feature = X_train_tensor[:, i]
            self.optimizer = torch.optim.Adam([weights], self.learning_rate)

            for j in range(self.n_opt_rounds):
                self.optimizer.zero_grad()
                # get train set error
                train_loss = self.__opt_func(X_segment=feature, y_segment=y_train_tensor, weights=weights)
                # append train loss to train loss history
                self.train_losses.append(train_loss.item())
                # create a train part of fit information
                train_output = f"train: {train_loss.item()}"
                # optimize weights according to the function
                if self.verbose != 0:
                    print("round:", j, train_output)
                train_loss.backward()
                self.optimizer.step()

            self.weights = torch.cat([self.weights, weights])

        self.weights = self.weights.reshape(-1, 2)
        self.fitted = True

    def get_weights(self) -> np.ndarray:
        if not self.fitted:
            raise AttributeError("Model has not been fitted yet. Use fit() method first.")

        return self.weights.detach().numpy()

    def transform(self, X) -> np.ndarray:
        X_tensor = to_tensor(X)
        if not self.fitted:
            raise AttributeError("Model has not been fitted yet. Use fit() method first.")

        return (X_tensor * self.weights[:, 0] + self.weights[:, 1]).detach().numpy()

