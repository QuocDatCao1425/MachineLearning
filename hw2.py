import numpy as np
import matplotlib.pyplot as plt

# Create some input / output data
x = np.array([0.03, 0.19, 0.34, 0.46, 0.78, 0.81, 1.08, 1.18, 1.39, 1.60, 1.65, 1.90])
y = np.array([0.67, 0.85, 1.05, 1.0 , 1.40, 1.5 , 1.3 , 1.54, 1.55, 1.68, 1.73, 1.6 ])

print(x)
print(y)

# Define 1D linear regression model
def f(x_, phi0, phi1):
	# TODO :  Replace this line with the linear regression model (eq 2.4)
	y_=phi0 + phi1*x_
	return y_

# Function to help plot the data
def plot(x, y, phi0, phi1):
    fig,ax = plt.subplots()
    ax.scatter(x,y)
    plt.xlim([0,2.0])
    plt.ylim([0,2.0])
    ax.set_xlabel('Input, $x$')
    ax.set_ylabel('Output, $y$')
    # Draw line
    x_line = np.arange(0,2,0.01)
    y_line = f(x_line, phi0, phi1)
    plt.plot(x_line, y_line,'b-',lw=2)

    plt.show()

# Set the intercept and slope as in figure 2.2b
phi0 = 0.4 ; phi1 = 0.2
# Plot the data and the model
plot(x,y,phi0,phi1)

# Function to calculate the loss
def compute_loss(x,y,phi0,phi1):

	# TODO Replace this line with the loss calculation (equation 2.5)
	loss=0
	for i in range(len(x)):
		loss=loss+(f(x[i],phi0, phi1)-y[i])**2
	return loss

# Compute the loss for our current model
loss = compute_loss(x,y,phi0,phi1)
print(f'Your Loss = {loss:3.2f}, Ground truth =7.07')

# Set the intercept and slope as in figure 2.2c
phi0 = 1.60 ; phi1 =-0.8
# Plot the data and the model
plot(x,y,phi0,phi1)
loss = compute_loss(x,y,phi0,phi1)
print(f'Your Loss = {loss:3.2f}, Ground truth =10.28')

# TO DO -- Change the parameters manually to fit the model
# First fix phi1 and try changing phi0 until you can't make the loss go down any more
# Then fix phi0 and try changing phi1 until you can't make the loss go down any more
# Repeat this process until you find a set of parameters that fit the model as in figure 2.2d
# You can either do this by hand, or if you want to get fancy, write code to descent automatically in this way
# Start at these values:
phi0 = 1.60 ; phi1 =-0.8

found_phi0=0.0
found_phi1=0.0
minimum_loss=9999999999999.9

while phi0 >= 0.0:
	phi1=-2.0
	while phi1 <= 2.0:
		loss=compute_loss(x,y,phi0,phi1)
		if loss<minimum_loss:
			found_phi0=phi0
			found_phi1=phi1
			minimum_loss=loss

		phi1=phi1+0.01
	phi0=phi0-0.01
		
phi0=found_phi0
phi1=found_phi1



plot(x,y,phi0,phi1)
print('phi0 =',phi0)
print('phi1 =',phi1)
print(f'Your Loss = {compute_loss(x,y,phi0,phi1):3.2f}')

# Make a 2D grid of possible phi0 and phi1 values
phi0_mesh, phi1_mesh = np.meshgrid(np.arange(0.0,2.0,0.02), np.arange(-1.0,1.0,0.02))

# Make a 2D array for the losses
all_losses = np.zeros_like(phi1_mesh)
# Run through each 2D combination of phi0, phi1 and compute loss
for indices,temp in np.ndenumerate(phi1_mesh):
    all_losses[indices] = compute_loss(x,y, phi0_mesh[indices], phi1_mesh[indices])


# Plot the loss function as a heatmap
fig = plt.figure()
ax = plt.axes()
fig.set_size_inches(7,7)
levels = 256
ax.contourf(phi0_mesh, phi1_mesh, all_losses ,levels)
levels = 40
ax.contour(phi0_mesh, phi1_mesh, all_losses ,levels, colors=['#80808080'])
ax.set_ylim([1,-1])
ax.set_xlabel(r'Intercept, $\phi_0$')
ax.set_ylabel(r'Slope, $\phi_1$')

# Plot the position of your best fitting line on the loss function
# It should be close to the minimum
ax.plot(phi0,phi1,'ro')
plt.show()
