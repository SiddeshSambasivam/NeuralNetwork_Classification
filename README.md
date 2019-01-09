# Neural Network: Classification
Implementing a neural network with one hidden layer to classify a planar data-set which gives a better practical intuition about the working of a NN. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## Prerequisites

What things you need to install the software and how to install them

```

Numpy
Matplotlib
tqdm
sklearn

```

## Installing

A step by step series of examples that tell you how to get a development env running

If the required dependencies (prerequisites) are already installed, please skip this part.
To install these prerequisites, Open the respective command line interface (powershell, terminal) and paste the following:

```
pip install numpy
pip install scikit-learn
pip install tqdm
pip install matplotlib

```
# What is a neural network?
Neural networks are algorithm that mimics the activity of brain. A neuron gets the input signals through dentrites ang gives the output signal through axon terminal. To replicate the neuron model, we fabricate an imaginary model which consist of several neuron model and form the neural network. In a single neuron model we have input wires with a bias which is passed to an acitvation function to get the required output.

![NN](https://i0.wp.com/blog.eyewire.org/wp-content/uploads/2015/12/a-cute-neuron-crashcourse.png?fit=1200%2C594&ssl=1)



# How does it work?
Every neural network will have one or more activation functions which determine the output from an neural unit, there are several activation functions such as sigmoid funcion, tanh function, RELU(Rectified linear unit), Leaky RELU.
Generally we give an input matrix with different features which is multiplied with a certain weights and added to the bias which is then passed to the activation function which spits the output to the next neuron and propogates to the end of the network to give the output, this in a nutshell is called the forward propogation.
After which the output is cross verified with the actual output and then the error is calculated, this error is called the loss which needs to be minimized. So we find out how much each component (weights, bias, activation outputs) need to change to give the desired output which again propogates all the way back to the input layer. Through this process the model gets trained efficiently though it could take time, moreover it is also computationally expensive so it is not recommened for problems which can be efficiently sovled by other machine learning algorithm.


### Network Architecture

![NN](https://github.com/IIplutocrat45II/NeuralNetwork_Classification/blob/master/classification_kiank.png)

Network architecture is the structure of the network, in other words we assign the number of hidden units and layers in a given neural network. Here we have 2 inputs (layer 0: As inputs are not considered as layer), 1 hidden layer with 4 hidden units and an output layer with one output.


# Quick Overview:
- [X] Defining a Neural network structure
- [x] Initializing the model parameters

Loop {
- [x] Implementing forword propogation algorithm        
- [x] Computeing the loss
- [x] Implementing back propogation algorithm
- [x] Updating the parameters by implementing gradient descent

}

### More Details
Clear description is given to each of the function in the python and the jupyter file.

Finally,Pull requests/changes/stars would be really helpful.
________________________________________________________________________________________________________________________

### Authored by
Siddesh S S 
