"""
TODO: implement a class that takes an array of windows of shape (N, w, d) and outputs pubs to be executed. Each pub must
 be the quantum circuit execution of a single window x = (x_{-w+1}, ..., x_0).
 Check what is the most efficient way to perform such computations in simulation mode: is it exact state_vector
 simulation then later on compute classical shadows in a direct randomized way or what! I guess so!
"""