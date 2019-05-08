"""
    Neural Network for Handwritten Digit Recognition
        Author:
            Connor Lambert (w910380)
        Under the Guidance of:
            Dr. Dia Ali
        Originally Done By:
            Michael Nielsen
        Updated for Python 3.5 By:
            Michał Dobrzański

        Shell Run Command:
            python test.py

        Command-Line Parameters:
        1st parameter is the Driver program, test.py
        2nd parameter is Epochs count
        3rd parameter is Batch size
        4th parameter is Learning Rate (eta)
"""

# ----------------------
# - Below we import the mnist_loader file, which loads in our database of digits

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

# ---------------------
# - Below we import the network file and create our network consisting of 784 Input Neurons, 30 Hidden Layer Neurons, and 10 Output Neurons
import network

net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data) # - Here we run our Stochastic Gradient Descent function to train our network
