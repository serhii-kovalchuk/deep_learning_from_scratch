import numpy as np

from sklearn.utils import shuffle
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import OneHotEncoder


class Network():
    
    def __init__(self, sizes, **kwargs):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]
        
        self.activation = kwargs.get("activation", "sigmoid")
        self.solver_name = kwargs.get("solver", "sgd")
        self.alpha = kwargs.get("alpha", 0.16)   # l2 regularization parameter
        
        self.encoder = OneHotEncoder(handle_unknown='ignore')
    
    def get_solver(self):
        try:
            return getattr(self, self.solver_name)
        except AttributeError:
            raise Exception(f"there is no such solver as {self.solver_name}")
    
    # to be compartible with sklearn
    def get_params(self, deep=True):
        return {
            "sizes": self.sizes,
            "activation": self.activation,
            "solver": self.solver_name,
            "alpha": self.alpha,
        }
    
     # to be compartible with sklearn
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def get_activation(self, z):
        return activation_functions[self.activation](z)
    
    def get_activation_dervitive(self, a):
        return activation_functions_derivitives[self.activation](a)
    
    def feedforward(self, a):
        a_set = [a]
        z_set = []
        
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            z_set.append(z)
            
            a = self.get_activation(z)
            a_set.append(a)
        
        return (z_set, a_set)
    
    def get_cost(self, y_predicted, y_true):
        return 0.5 * (y_predicted - y_true)**2
    
    def get_cost_derivitive(self, y_predicted, y_true):
        return y_predicted - y_true
    
    def get_weights_and_biases_gradients_for_layer(
            self, layer_i, deltas, activations):
        
        gradient_b = deltas.sum(1).reshape((-1, 1))
        gradient_w = np.dot(deltas, activations.T) + \
            self.alpha * self.weights[layer_i]   # regularization
        
        return gradient_b, gradient_w
    
    def backpropagate(self, x_train, y_train, etha):
        training_instances_number = y_train.shape[0]
        
        gradient_w = [0] * len(self.weights)
        gradient_b = [0] * len(self.biases)
        
        # forward pass
        z_set, a_set = self.feedforward(x_train.T)
        
        # backpropagation
        deltas = self.get_cost_derivitive(a_set[-1], y_train.T) * \
            self.get_activation_dervitive(a_set[-1])
        # is an array of gradient vectors of the cost function 
        # with respect to the only current layer input values z
        
        gradient_b[-1], gradient_w[-1] = \
            self.get_weights_and_biases_gradients_for_layer(
                -1, deltas, a_set[-2])
        
        for i in range(self.num_layers-2, 0, -1):
            layer_size = self.sizes[i]
            
            deltas = np.dot(self.weights[i].T, deltas) * \
                self.get_activation_dervitive(a_set[i])
            
            gradient_b[i-1], gradient_w[i-1] = \
                self.get_weights_and_biases_gradients_for_layer(
                    i-1, deltas, a_set[i-1])
        
        return (
            np.array(gradient_w) / training_instances_number, 
            np.array(gradient_b) / training_instances_number
        )
    
    def sgd(self, x_train, y_train, **kwargs):
        etha = kwargs.get("etha", 3.0)
        n_epoches = kwargs.get("n_epoches", 7000)
        minibatch_size = kwargs.get("minibatch_size", y_train.shape[0])
        
        n = len(x_train)
        
        for i in range(n_epoches):
            x_batch, y_batch = shuffle(
                x_train, 
                y_train, 
                n_samples=minibatch_size
            )

            # gradient with respect to weights and biases
            gradient_w, gradient_b = self.backpropagate(
                x_batch, 
                y_batch, 
                etha
            )
            
            self.weights = [w - etha*gw for w, gw in zip(self.weights, gradient_w)]
            self.biases = [b - etha*gb for b, gb in zip(self.biases, gradient_b)]
    
    def newton_method(self, x_train, y_train, **kwargs):
        raise NotImplementedError("solver is not implemented")
    
    def encode_y_classes(self, y_data):
        return self.encoder.fit_transform(y_data.reshape(-1, 1)).toarray()
    
    def decode_y_classes(self, y_data_encoded):
        return self.encoder.inverse_transform(y_data_encoded).reshape(-1)
    
    def predict(self, x_data):
        n_classes = len(self.encoder.categories_[0])
        y_predicted = []
        
        for x in x_data:
            a = x.reshape(-1, 1)
            for b, w in zip(self.biases, self.weights):
                a = self.get_activation(np.dot(w, a)+b)
            
            a_arg_max = np.argmax(a)
            y_prediction = np.zeros(n_classes)
            y_prediction[a_arg_max] = 1
            y_predicted.append(y_prediction)
        
        return self.decode_y_classes(y_predicted)
    
    def score(self, x_data, y_data):
        if x_data.shape[0] != y_data.shape[0]:
            raise Exception("x- and y-data should be of equal length")
        
        y_predicted = self.predict(x_data)
        correct_predictions = 0
        
        for y_prediction, y in zip(y_predicted, y_data):
            if y_prediction == y:
                correct_predictions += 1
        
        return correct_predictions / y_data.shape[0]
    
    def fit(self, x_train, y_train, **kwargs):
        solver = self.get_solver()
        solver(x_train, self.encode_y_classes(y_train), **kwargs)
        
        return self.score(x_train, y_train)


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def sigmoid_derivitive(sigmoid_z):
    return sigmoid_z * (1 - sigmoid_z)


infinity = float("inf")

def relu(z):
    max_value = infinity
    z_zeros = np.zeros(z.shape)
    # return np.maximum(z_zeros, z)
    z_max = np.full(z.shape, 1)
    return np.maximum(z_zeros, np.minimum(z, z_max))


def relu_derivitive(relu_z):
    relu_z[relu_z > 0] = 1
    
    return relu_z


def tanh(z):
    return np.tanh(z)


def tanh_derivitive(tanh_z):
    return 1 - tanh_z ** 2


activation_functions = {
    "relu": relu,
    "tanh": tanh,
    "sigmoid": sigmoid
}

activation_functions_derivitives = {
    "relu": relu_derivitive,
    "tanh": tanh_derivitive,
    "sigmoid": sigmoid_derivitive
}
