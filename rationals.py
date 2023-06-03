import json

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RationalsModel(nn.Module):
    # n = batch size-1 (svhn 64 /tiny 256), m = number of hidden layers
    def __init__(self, n=63, m=64, function="relu", use_coefficients=False):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the attributes of an instance of a class.


        :param self: Represent the instance of the class
        :param n: Set the number of inputs to the network
        :param m: Define the number of hidden layers
        :param function: Specify the activation function
        :param use_coefficients: Determine if the coefficients should be loaded
        from a file or not
        :return: Nothing
        """
        super(RationalsModel, self).__init__() #mod pytorch
        self.n = n
        self.m = m
        self.p = n + m
        self.func_name = function
        self.function = self.load_func(func=function)
        if use_coefficients:
            self.load_coefficients()
        else:
            self.coefficients = torch.randn(self.p + 1, requires_grad=True)

        self.function_list = ["relu", "sigmoid", "tanh", "leaky_relu", "swish", "gelu"]

        self.store_coefficients()

    def forward(self, x):
        """
        The forward function computes the polynomial given by:

        :param self: Represent the instance of the class
        :param x: Compute the polynomial
        :return: A tensor of shape (n_samples, 1)
        """
        x_tensor = x.clone().detach().to(device)
        #device = x_tensor.device
        
        
        n_coeffs = self.coefficients[: self.n + 1]
        m_coeffs = self.coefficients[self.n + 1 :]

        x_powers = torch.pow(x_tensor, torch.arange(self.n + 1, device=device).unsqueeze(1))
        numerator = torch.matmul(n_coeffs.to(device), x_powers)

        x_powers = torch.pow(x_tensor, torch.arange(1, self.m + 1, device=device).unsqueeze(1))
        denominator = torch.matmul(m_coeffs.to(device), x_powers)

        # compute overall polynomial
        polynomial = numerator / (torch.abs(denominator) + 1)

        return polynomial

    def train(
        self, name, x_train, y_train, learning_rate=0.01, num_epochs=10000, render=False
    ):
        """
        The train function takes in a name, x_train, y_train, learning rate (default 0.01),
        number of epochs (default 10000), and render flag (default False). The function then
        creates an optimizer using the Adam optimizer with the coefficients as parameters and
        a loss function using MSE. It also initializes some lists to store values for plotting
        later on. The update plot function is defined here as well which will be used by
        FuncAnimation to animate our plots later on. Then we loop through each epoch and
        calculate our predicted y value based off of our current coefficients from forward().

        :param self: Refer to the class instance itself
        :param name: Save the trained model to a file
        :param x_train: Train the model
        :param y_train: Train the model to predict a function
        :param learning_rate: Control the speed of learning
        :param num_epochs: Set the number of iterations that the model will train for
        :param render: Render the animation of the training process
        :return: The loss of the last epoch
        """
        optimizer = torch.optim.Adam([self.coefficients], lr=learning_rate)
        loss_fn = torch.nn.MSELoss()
        end_loss = 100.0

        fig, ax = plt.subplots()
        y_pred_values = []  # Store y_pred values for plotting
        epoch_values = []  # Store epoch values for plotting
        predictions = []
        true_functions = []

        def update_plot(epoch):
            ax.clear()
            ax.plot(x_train.numpy(), predictions[epoch], label=f"predict {name}")
            ax.plot(x_train.numpy(), true_functions[epoch], label=f"true {name}")
            ax.legend()
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"Epoch {epoch_values[epoch]}")

        for epoch in range(num_epochs):
            y_pred = self.forward(x_train)
            loss = loss_fn(y_pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
                predictions.append(y_pred.detach().numpy())
                true_functions.append(y_train.numpy())
                epoch_values.append(epoch)

            end_loss = loss.item()
            if end_loss > 10:
                break

        ani = animation.FuncAnimation(
            fig, update_plot, frames=len(epoch_values), interval=1
        )
        if render:
            ani.save(
                f"rational_trained_models/{name}_function.gif",
                writer="imagemagick",
                fps=30,
            )
            plt.close()

        return end_loss

    def load_func(self, func="relu", space=None):
        """
        The load_func function takes in a string and returns the corresponding
        activation function.
            Args:
                func (str): The name of the activation function to be returned.

        :param self: Represent the instance of the class
        :param func: Specify the activation function to be used
        :param space: Define the range of values that will be passed to the
        activation function
        :return: The activation function that is passed to it
        """
        if space is None:
            space = torch.linspace(-2, 2, 100)
        if func == "relu":
            true_func = torch.nn.functional.relu(space)
        elif func == "sigmoid":
            true_func = torch.nn.functional.sigmoid(space)
        elif func == "tanh":
            true_func = torch.nn.functional.tanh(space)
        elif func == "leaky_relu":
            true_func = torch.nn.functional.leaky_relu(space)
        elif func == "swish":
            true_func = torch.nn.functional.silu(space)
        elif func == "gelu":
            true_func = torch.nn.functional.gelu(space)
        else:
            true_func = torch.nn.functional.relu(space)
        return true_func

    def _show(self, x_true, y_true, func=None):
        """
        The _show function is a helper function that plots the true and
        predicted functions. It takes in three arguments: x_true, y_true,
        and func. The first two are the data points that we want to plot
        on our graph (the true values). The third argument is optional; it's
        a string that will be used as a label for the legend of our graph.

        :param self: Access the attributes and methods of the class
        :param x_true: Plot the true function
        :param y_true: Plot the true function
        :param func: Label the plot
        :return: A plot of the true function and the predicted function
        """
        x_pred = torch.linspace(-2, 2, 1000)
        y_pred = self.forward(x_pred)

        plt.plot(x_true, y_true, label=f"True {func}")
        plt.plot(x_pred, y_pred.detach().numpy(), label=f"Predicted {func}")
        plt.legend()
        plt.show()

    def show_all(self):
        """
        The show_all function takes in a list of functions and plots the
        true function, the trained model's approximation, and the error
        between them. It also prints out the mean squared error for each
        function.

        :param self: Access the attributes of the class
        :return: A plot of the functions and their approximations
        """
        x_true = torch.linspace(-2, 2, 100)

        for func in self.function_list:
            name = f"rational_trained_models/{func}_model.pt"
            model = torch.load(name)
            y_true = model.load_func(func)
            model._show(x_true, y_true, func)

    def show(self):
        """
        The show function takes the trained model and plots it against
        the true function. It also prints out a table of values for both
        functions at various points.

        :param self: Access the attributes and methods of the class
        :return: A plot of the function, and the predicted values
        """
        x_true = torch.linspace(-2, 2, 100)
        func = self.func_name
        name = f"rational_trained_models/{func}_model.pt"
        model = torch.load(name)
        y_true = model.load_func(func)
        model._show(x_true, y_true, func)

    def store_coefficients(self):
        """
        The store_coefficients function takes the coefficients of a rational function
        and stores them in a JSON file. The name of the JSON file is based on the name
        of the function that was used to generate it. This allows us to store multiple
        trained models for different functions, and then load them later.

        :param self: Refer to the object itself
        :return: A dictionary with the function name and coefficients
        """
        coefficients_dict = {
            "function": self.func_name,
            "coefficients": self.coefficients.tolist(),
        }
        with open(f"rational_trained_models/coeff_{self.func_name}.json", "w") as file:
            json.dump(coefficients_dict, file)

    def load_coefficients(self):
        """
        The load_coefficients function loads the coefficients of a trained model
        from a json file. The function takes no arguments and returns nothing.
        The function will raise an error if the  json file does not exist or if
        it contains data for another function.

        :param self: Refer to the object itself
        :return: The coefficients for the function name given
        """
        try:
            with open(
                f"rational_trained_models/coeff_{self.func_name}.json", "r"
            ) as file:
                data = json.load(file)
                if data["function"] == self.func_name:
                    self.coefficients = torch.tensor(
                        data["coefficients"], requires_grad=True
                    )
                else:
                    raise ValueError("No function found!")
        except FileNotFoundError:
            print("No file found!")


def train_all(render=True, epsilon=0.001, use_coefficients=False):
    """
    The train_all function trains a rational function for each of the activation
    functions in the function_list. The model is trained until it reaches an end
    loss of epsilon, which defaults to 0.001.

    :param render: Determine whether or not to show the plot of the model's progress
    :param epsilon: Determine when to stop training the model
    :param use_coefficients: Determine whether the coefficients of the rational function are
    :return: Nothing
    """
    function_list = ["relu", "sigmoid", "tanh", "leaky_relu", "swish", "gelu"]

    for func in function_list:
        model = RationalsModel(
            n=5, m=5, function=func, use_coefficients=use_coefficients
        )
        x_train = torch.linspace(-2, 2, 100)
        if func == "relu":
            true_func = torch.nn.functional.relu(x_train)
        elif func == "sigmoid":
            true_func = torch.nn.functional.sigmoid(x_train)
        elif func == "tanh":
            true_func = torch.nn.functional.tanh(x_train)
        elif func == "leaky_relu":
            true_func = torch.nn.functional.leaky_relu(x_train)
        elif func == "swish":
            true_func = torch.nn.functional.silu(x_train)
        elif func == "gelu":
            true_func = torch.nn.functional.gelu(x_train)
        else:
            print("Invalid function name: %s" % func)

        y_train = true_func

        end_loss = model.train(func, x_train, y_train, render=render)

        while end_loss > epsilon:
            model = RationalsModel(
                n=5, m=5, function=func, use_coefficients=use_coefficients
            )
            y_train = true_func
            end_loss = model.train(func, x_train, y_train, render=render)

        torch.save(model, f"rational_trained_models/{func}_model.pt")


if __name__ == "__main__":
    # train_all(render=True, use_coefficients=True)

    # plot results
    # mod = RationalsModel(function="leaky_relu")
    mod = RationalsModel()
    # mod.show()
    mod.show_all()
