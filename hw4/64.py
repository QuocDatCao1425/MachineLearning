# import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

# Let's create our training data of 30 pairs {x_i, y_i}
data = np.array([[-1.920e+00,-1.422e+01,1.490e+00,-1.940e+00,-2.389e+00,-5.090e+00,
                 -8.861e+00,3.578e+00,-6.010e+00,-6.995e+00,3.634e+00,8.743e-01,
                 -1.096e+01,4.073e-01,-9.467e+00,8.560e+00,1.062e+01,-1.729e-01,
                  1.040e+01,-1.261e+01,1.574e-01,-1.304e+01,-2.156e+00,-1.210e+01,
                 -1.119e+01,2.902e+00,-8.220e+00,-1.179e+01,-8.391e+00,-4.505e+00],
                  [-1.051e+00,-2.482e-02,8.896e-01,-4.943e-01,-9.371e-01,4.306e-01,
                  9.577e-03,-7.944e-02 ,1.624e-01,-2.682e-01,-3.129e-01,8.303e-01,
                  -2.365e-02,5.098e-01,-2.777e-01,3.367e-01,1.927e-01,-2.222e-01,
                  6.352e-02,6.888e-03,3.224e-02,1.091e-02,-5.706e-01,-5.258e-02,
                  -3.666e-02,1.709e-01,-4.805e-02,2.008e-01,-1.904e-01,5.952e-01]])

# Define our model
def model(phi,x):
  sin_component = np.sin(phi[0] + 0.06 * phi[1] * x)
  gauss_component = np.exp(-(phi[0] + 0.06 * phi[1] * x) * (phi[0] + 0.06 * phi[1] * x) / 32)
  y_pred = sin_component * gauss_component
  return y_pred

# Draw model
def draw_model(data, model, phi, title=None):
    x_model = np.arange(-15, 15, 0.1)
    y_model = model(phi, x_model)

    fig, ax = plt.subplots()
    ax.plot(data[0,:], data[1,:], 'bo')
    ax.plot(x_model, y_model, 'm-')
    ax.set_xlim([-15, 15])
    ax.set_ylim([-1, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if title is not None:
        ax.set_title(title)
    plt.show()

# Initialize the parameters and draw the model
phi = np.zeros((2,1))
phi[0] = -5     # Horizontal offset
phi[1] = 25     # Frequency
draw_model(data, model, phi, "Initial parameters")

# Compute Loss
def compute_loss(data_x, data_y, model, phi):
    pred_y = model(phi, data_x)
    loss = np.sum((pred_y - data_y)**2)
    return loss

# Draw loss function
def draw_loss_function(compute_loss, data, model, phi_iters=None):
    # Define pretty colormap
    my_colormap_vals_hex = ('2a0902', '2b0a03', '2c0b04', '2d0c05', '2e0c06', '2f0d07', '300d08', '310e09', '320f0a', '330f0b')
    my_colormap_vals_dec = np.array([int(element, base=16) for element in my_colormap_vals_hex])
    r = np.floor(my_colormap_vals_dec / (256*256))
    g = np.floor((my_colormap_vals_dec - r * 256 * 256) / 256)
    b = np.floor(my_colormap_vals_dec - r * 256 * 256 - g * 256)
    my_colormap = ListedColormap(np.vstack((r, g, b)).transpose() / 255.0)

    # Make grid of offset/frequency values to plot
    offsets_mesh, freqs_mesh = np.meshgrid(np.arange(-10, 10.0, 0.1), np.arange(2.5, 22.5, 0.1))
    loss_mesh = np.zeros_like(freqs_mesh)

    # Compute loss for every set of parameters
    for idslope, slope in np.ndenumerate(freqs_mesh):
        loss_mesh[idslope] = compute_loss(data[0,:], data[1,:], model, np.array([[offsets_mesh[idslope]], [slope]]))

    fig, ax = plt.subplots()
    fig.set_size_inches(8,8)
    ax.contourf(offsets_mesh, freqs_mesh, loss_mesh, 256, cmap=my_colormap)
    ax.contour(offsets_mesh, freqs_mesh, loss_mesh, 20, colors=['#80808080'])
    if phi_iters is not None:
        ax.plot(phi_iters[0,:], phi_iters[1,:],'go-')
    ax.set_ylim([2.5, 22.5])
    ax.set_xlabel('Offset $\phi_{0}$')
    ax.set_ylabel('Frequency, $\phi_{1}$')
    plt.show()

draw_loss_function(compute_loss, data, model)

# Derivative for Gabor model with respect to phi0 and phi1
def gabor_deriv_phi0(data_x, data_y, phi0, phi1):
    x = 0.06 * phi1 * data_x + phi0
    y = data_y
    cos_component = np.cos(x)
    sin_component = np.sin(x)
    gauss_component = np.exp(-0.5 * x**2 / 16)
    deriv = cos_component * gauss_component - sin_component * gauss_component * x / 16
    deriv = 2 * deriv * (sin_component * gauss_component - y)
    return np.sum(deriv)

def gabor_deriv_phi1(data_x, data_y, phi0, phi1):
    x = 0.06 * phi1 * data_x + phi0
    y = data_y
    cos_component = np.cos(x)
    sin_component = np.sin(x)
    gauss_component = np.exp(-0.5 * x**2 / 16)
    deriv = 0.06 * data_x * cos_component * gauss_component - 0.06 * data_x * sin_component * gauss_component * x / 16
    deriv = 2 * deriv * (sin_component * gauss_component - y)
    return np.sum(deriv)

# Compute Gradient
def compute_gradient(data_x, data_y, phi):
    dl_dphi0 = gabor_deriv_phi0(data_x, data_y, phi[0], phi[1])
    dl_dphi1 = gabor_deriv_phi1(data_x, data_y, phi[0], phi[1])
    return np.array([[dl_dphi0], [dl_dphi1]])

# Set the random number generator so you always get the same numbers (disable if you don't want this)
np.random.seed(1)
n_steps = 81
batch_size = 5
alpha = 0.6

# Initialize parameters and momentum
phi_all = np.zeros((2, n_steps+1))
phi_all[0,0] = -1.5
phi_all[1,0] = 6.5

# Measure loss and draw the initial model
loss = compute_loss(data[0,:], data[1,:], model, phi_all[:,0:1])
draw_model(data, model, phi_all[:,0:1], "Initial parameters, Loss = %f" % loss)

# Gradient descent loop
for c_step in range(n_steps):
    # Choose random batch indices
    batch_index = np.random.permutation(data.shape[1])[0:batch_size]
    # Compute the gradient
    gradient = compute_gradient(data[0, batch_index], data[1, batch_index], phi_all[:,c_step:c_step+1])
    # Update the parameters
    phi_all[:,c_step+1:c_step+2] = phi_all[:,c_step:c_step+1] - alpha * gradient

    # Measure loss and draw model for each step
    loss = compute_loss(data[0,:], data[1,:], model, phi_all[:,c_step+1:c_step+2])
    draw_model(data, model, phi_all[:,c_step+1], "Iteration %d, loss = %f" % (c_step+1, loss))

# Draw the trajectory on the loss function
draw_loss_function(compute_loss, data, model, phi_all)
