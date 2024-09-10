# Imports math library
import numpy as np
# Imports plotting library
import matplotlib.pyplot as plt

# Define the Rectified Linear Unit (ReLU) function
def ReLU(preactivation):
  # TODO write code to implement the ReLU and compute the activation at the
  # hidden unit from the preactivation
  # This should work on every element of the ndarray "preactivation" at once
  # One way to do this is with the ndarray "clip" function
  # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.clip.html

  activation = np.maximum(0,preactivation);

  return activation

# Make an array of inputs
z = np.arange(-5,5,0.1)
RelU_z = ReLU(z)

# Plot the ReLU function
fig, ax = plt.subplots()
ax.plot(z,RelU_z,'r-')
ax.set_xlim([-5,5]);ax.set_ylim([-5,5])
ax.set_xlabel('z'); ax.set_ylabel('ReLU[z]')
ax.set_aspect('equal')
plt.show()

# Define a shallow neural network with, one input, one output, and three hidden units
def shallow_1_1_3(x, activation_fn, phi_0, phi_1, phi_2, phi_3, theta_10, theta_11, theta_20, theta_21, theta_30, theta_31):
    # Compute the preactivations
    pre_1 = theta_10 + theta_11 * x  # Pre-activation for the first hidden unit
    pre_2 = theta_20 + theta_21 * x  # Pre-activation for the second hidden unit
    pre_3 = theta_30 + theta_31 * x  # Pre-activation for the third hidden unit

    # Compute the activations using the activation function (e.g., ReLU)
    act_1 = activation_fn(pre_1)  # Activation for the first hidden unit
    act_2 = activation_fn(pre_2)  # Activation for the second hidden unit
    act_3 = activation_fn(pre_3)  # Activation for the third hidden unit

    # Compute the weighted activations (hidden layer to output layer)
    w_act_1 = phi_1 * act_1  # Weight the first hidden activation
    w_act_2 = phi_2 * act_2  # Weight the second hidden activation
    w_act_3 = phi_3 * act_3  # Weight the third hidden activation

    # Combine the weighted activations and add bias phi_0 to get the output
    y = phi_0 + w_act_1 + w_act_2 + w_act_3

    # Return all intermediate and final values
    return y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3

# Plot the shallow neural network.  We'll assume input in is range [0,1] and output [-1,1]
# If the plot_all flag is set to true, then we'll plot all the intermediate stages as in Figure 3.3
def plot_neural(x, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3, plot_all=False, x_data=None, y_data=None):

  # Plot intermediate plots if flag set
  if plot_all:
    fig, ax = plt.subplots(3,3)
    fig.set_size_inches(8.5, 8.5)
    fig.tight_layout(pad=3.0)
    ax[0,0].plot(x,pre_1,'r-'); ax[0,0].set_ylabel('Preactivation')
    ax[0,1].plot(x,pre_2,'b-'); ax[0,1].set_ylabel('Preactivation')
    ax[0,2].plot(x,pre_3,'g-'); ax[0,2].set_ylabel('Preactivation')
    ax[1,0].plot(x,act_1,'r-'); ax[1,0].set_ylabel('Activation')
    ax[1,1].plot(x,act_2,'b-'); ax[1,1].set_ylabel('Activation')
    ax[1,2].plot(x,act_3,'g-'); ax[1,2].set_ylabel('Activation')
    ax[2,0].plot(x,w_act_1,'r-'); ax[2,0].set_ylabel('Weighted Act')
    ax[2,1].plot(x,w_act_2,'b-'); ax[2,1].set_ylabel('Weighted Act')
    ax[2,2].plot(x,w_act_3,'g-'); ax[2,2].set_ylabel('Weighted Act')

    for plot_y in range(3):
      for plot_x in range(3):
        ax[plot_y,plot_x].set_xlim([0,1]);ax[plot_x,plot_y].set_ylim([-1,1])
        ax[plot_y,plot_x].set_aspect(0.5)
      ax[2,plot_y].set_xlabel('Input, $x$');
    plt.show()

  fig, ax = plt.subplots()
  ax.plot(x,y)
  ax.set_xlabel('Input, $x$'); ax.set_ylabel('Output, $y$')
  ax.set_xlim([0,1]);ax.set_ylim([-1,1])
  ax.set_aspect(0.5)
  if x_data is not None:
    ax.plot(x_data, y_data, 'mo')
    for i in range(len(x_data)):
      ax.plot(x_data[i], y_data[i],)
  plt.show()
# Now lets define some parameters and run the neural network
theta_10 =  0.3 ; theta_11 = -1.0
theta_20 = -1.0  ; theta_21 = 2.0
theta_30 = -0.5  ; theta_31 = 0.65
phi_0 = -0.3; phi_1 = 2.0; phi_2 = -1.0; phi_3 = 7.0

# Define a range of input values
x = np.arange(0,1,0.01)

# We run the neural network for each of these input values
y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3 = \
    shallow_1_1_3(x, ReLU, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11, theta_20, theta_21, theta_30, theta_31)
# And then plot it
plot_neural(x, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3, plot_all=True)

# TODO
# 1. Predict what effect changing phi_0 will have on the network.
# Changing phi_0 will shift the entire output y up or down by the value of phi_0

# 2. Predict what effect multiplying phi_1, phi_2, phi_3 by 0.5 would have.  Check if you are correct
# It will scale down the contributions of the activations from each hidden unit. As a result, the output y will also be scaled down by 0.5, effectively shrinking the output curve vertically.

# 3. Predict what effect multiplying phi_1 by -1 will have.  Check if you are correct.
# It will invert the contribution from the first hidden unit. If the first hidden unit was contributing positively, it will contribute negatively, which may flip the curve or change its shape based on how the other terms are weighted.

# 4. Predict what effect setting theta_20 to -1.2 will have.  Check if you are correct.
# It  will cause the activation of the second hidden unit to shift, likely reducing its activation for the same input values and modifying the overall output

# 5. Change the parameters so that there are only two "joints" (including outside the range of the plot)
# Set one of the hidden activations to a constant, by adjusting theta_ij values, so it doesn’t vary with the input.
# There are actually three ways to do this. See if you can figure them all out
# Set one of the phi values to 0. For example, if phi_3 = 0, the third hidden unit’s contribution would be eliminated.
# Set one of the weights, such as theta_11, theta_21, or theta_31, to 0. This would essentially deactivate one of the hidden units, making the output depend on only two hidden units.

# 6. With the original parameters, the second line segment is flat (i.e. has slope zero)
# How could you change theta_10 so that all of the segments have non-zero slopes
# The first hidden unit’s preactivation equation is influenced by theta_10 and theta_11. If the current setup results in one flat segment, adjusting theta_10 (or theta_11) can create non-zero slopes.

# 7. What do you predict would happen if you multiply theta_20 and theta21 by 0.5, and phi_2 by 2.0?
# Check if you are correct.
# Multiplying theta_20 and theta_21 by 0.5 will reduce the slope and bias of the second hidden unit’s preactivation, resulting in smaller activations for the same input values.
# However, multiplying phi_2 by 2.0 will increase the contribution of the second hidden unit in the output.

# 8. What do you predict would happen if you multiply theta_20 and theta21 by -0.5, and phi_2 by -2.0?
# Check if you are correct.
# Multiplying theta_20 and theta_21 by -0.5 will invert the direction of the second hidden unit’s contribution, making it act in the opposite direction for increasing inputs.
# Multiplying phi_2 by -2.0 will amplify this inverted contribution, likely creating a drastic shift in the output curve, where the second hidden unit's influence will be inverted and amplified.

theta_10 =  0.3 ; theta_11 = -1.0
theta_20 = -0.6  ; theta_21 = 1.0
theta_30 = -0.5  ; theta_31 = 0.65
phi_0 = -0.3; phi_1 = -1.0; phi_2 = -1.0; phi_3 = 3.5

# Define a range of input values
x = np.arange(0,1,0.01)

# We run the neural network for each of these input values
y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3 = \
    shallow_1_1_3(x, ReLU, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11, theta_20, theta_21, theta_30, theta_31)
# And then plot it
plot_neural(x, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3, plot_all=True)

# Least squares function
def least_squares_loss(y_train, y_predict):
  # TODO Replace the line below to compute the sum of squared
  # differences between the real values of y and the predicted values from the model f[x_i,phi]
  # (see figure 2.2 of the book)
  # you will need to use the function np.sum
  loss = np.sum((y_train - y_predict) ** 2)

  return loss

# Now lets define some parameters, run the neural network, and compute the loss
theta_10 =  0.3 ; theta_11 = -1.0
theta_20 = -1.0  ; theta_21 = 2.0
theta_30 = -0.5  ; theta_31 = 0.65
phi_0 = 0.2; phi_1 = -1.0; phi_2 = -1.0; phi_3 = 7.0

# Define a range of input values
x = np.arange(0,1,0.01)

x_train = np.array([0.09291784,0.46809093,0.93089486,0.67612654,0.73441752,0.86847339,\
                   0.49873225,0.51083168,0.18343972,0.99380898,0.27840809,0.38028817,\
                   0.12055708,0.56715537,0.92005746,0.77072270,0.85278176,0.05315950,\
                   0.87168699,0.58858043])
y_train = np.array([-0.15934537,0.18195445,0.451270150,0.13921448,0.09366691,0.30567674,\
                    0.372291170,0.40716968,-0.08131792,0.41187806,0.36943738,0.3994327,\
                    0.019062570,0.35820410,0.452564960,-0.0183121,0.02957665,-0.24354444, \
                    0.148038840,0.26824970])

# We run the neural network for each of these input values
y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3 = \
    shallow_1_1_3(x, ReLU, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11, theta_20, theta_21, theta_30, theta_31)
# And then plot it
plot_neural(x, y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3, plot_all=True, x_data = x_train, y_data = y_train)

# Run the neural network on the training data
y_predict, *_ = shallow_1_1_3(x_train, ReLU, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11, theta_20, theta_21, theta_30, theta_31)

# Compute the least squares loss and print it out
loss = least_squares_loss(y_train,y_predict)
print('Your Loss = %3.3f, True value = 9.385'%(loss))

# TODO.  Manipulate the parameters (by hand!) to make the function
# fit the data better and try to reduce the loss to as small a number
# as possible.  The best that I could do was 0.181
# Tip... start by manipulating phi_0.
# It's not that easy, so don't spend too much time on this!
