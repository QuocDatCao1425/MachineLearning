import numpy as np
import matplotlib.pyplot as plt
def linear_function_1D(x, beta, omega):
	y=beta+x*omega
	return y

x = np.arange(0.0,10.0, 0.01)
beta = 0.0; omega = 1.0
y = linear_function_1D(x,beta,omega)

def graphing(x,y):
	fig, ax = plt.subplots()
	ax.plot(x,y,'r-')
	ax.set_ylim([0,10]);ax.set_xlim([0,10])
	ax.set_xlabel('x'); ax.set_ylabel('y')
	plt.show()

# TODO -- experiment with changing the values of beta and omega
beta=10
omega=-2
y = linear_function_1D(x,beta,omega)
graphing(x,y)

def draw_2D_function(x1_mesh, x2_mesh, y):
    fig, ax = plt.subplots()
    fig.set_size_inches(7,7)
    pos = ax.contourf(x1_mesh, x2_mesh, y, levels=256 ,cmap = 'hot', vmin=-10,vmax=10.0)
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel('x1');ax.set_ylabel('x2')
    levels = np.arange(-10,10,1.0)
    ax.contour(x1_mesh, x2_mesh, y, levels, cmap='winter')
    plt.show()

def linear_function_2D(x1,x2,beta,omega1,omega2):
	# TODO -- replace the code line below with formula for 2D linear equation
	y = beta + x1*omega1 + x2*omega2
	return y
x1 = np.arange(0.0, 10.0, 0.1)
x2 = np.arange(0.0, 10.0, 0.1)
x1,x2 = np.meshgrid(x1,x2)

# Compute the 2D function for given values of omega1, omega2
beta = 0.0; omega1 = 1.0; omega2 = -0.5
y  = linear_function_2D(x1,x2,beta, omega1, omega2)

# Draw the function.
# Color represents y value (brighter = higher value)
# Black = -10 or less, White = +10 or more
# 0 = mid orange
# Lines are contours where value is equal
draw_2D_function(x1,x2,y)

# TODO
# Predict what this plot will look like if you set omega_1 to zero
# ---> The line will be horizontal and the color will look darker because omega1*x1=0 and omega = -0.5 cause the y values smaller than the original one
# Change the code and see if you are right.
omega1 = 0.0
y  = linear_function_2D(x1,x2,beta, omega1, omega2)
draw_2D_function(x1,x2,y)

# TODO
# Predict what this plot will look like if you set omega_2 to zero
# ---> the line will be in vertical and The color will look brighter because omega1*x1= 1 and omega = 0 cause the y values bigger than the original one. 
# Change the code and see if you are right
omega1 = 1.0
omega2 = 0.0
y  = linear_function_2D(x1,x2,beta, omega1, omega2)
draw_2D_function(x1,x2,y)

# TODO
# Predict what this plot will look like if you set beta to -5
# ---> The graph look darker because y value get smaller by 5
# Change the code and see if you are correct
omega1 = 1.0
omega2 = -0.5
beta = -5
y  = linear_function_2D(x1,x2,beta, omega1, omega2)
draw_2D_function(x1,x2,y)

#----------------------------------------------------------------------------------------------------------------
# Define a linear function with three inputs, x1, x2, and x_3
def linear_function_3D(x1,x2,x3,beta,omega1,omega2,omega3):
	# TODO -- replace the code below with formula for a single 3D linear equation
	y = beta + omega1*x1 + omega2*x2 + omega3*x3

	return y

# Define the parameters
beta1 = 0.5; beta2 = 0.2
omega11 =  -1.0 ; omega12 = 0.4; omega13 = -0.3
omega21 =  0.1  ; omega22 = 0.1; omega23 = 1.2

# Define the inputs
x1 = 4 ; x2 =-1; x3 = 2

# Compute using the individual equations
y1 = linear_function_3D(x1,x2,x3,beta1,omega11,omega12,omega13)
y2 = linear_function_3D(x1,x2,x3,beta2,omega21,omega22,omega23)
print("Individual equations")
print('y1 = %3.3f\ny2 = %3.3f'%((y1,y2)))

# Define vectors and matrices
beta_vec = np.array([[beta1],[beta2]])
omega_mat = np.array([[omega11,omega12,omega13],[omega21,omega22,omega23]])
x_vec = np.array([[x1], [x2], [x3]])

# Compute with vector/matrix form
y_vec = beta_vec+np.matmul(omega_mat, x_vec)
print("Matrix/vector form")
print('y1= %3.3f\ny2 = %3.3f'%((y_vec[0][0],y_vec[1][0])))

# ------>>>>>>>>QUESTION
# A single linear equation with three inputs (i.e. linear_function_3D()) associates a value y with each point in a 3D space (x1,x2,x3). Is it possible to visualize this? What value is at position (0,0,0)?
# It is possible to visualize this three point in 3D space create a a plan. the y value can be color or height, etc. associated with the plan.
# at y(0,0,0)=beta.

#Write code to compute three linear equations with two inputs (x1,x2) using both the individual equations and the matrix form (you can make up any values for the inputs beta and the slopes omega
beta1 = 3.0
beta2 = 4.0
beta3 = 5.0
omega11 = 1.0; omega12 = 2.0
omega21 = 3.0; omega22 = 4.0
omega31 = 5.0; omega32 = 6.0
def individual_eq(x1,x2):
	y1=linear_function_2D(x1,x2,beta1,omega11,omega12)
	y2=linear_function_2D(x1,x2,beta2,omega21,omega22)
	y3=linear_function_2D(x1,x2,beta3,omega31,omega32)
	Y=np.array([[y1],[y2],[y3]])
	return Y
def matrix_eq(x1,x2):
	O=np.array([
		[omega11,omega12],
		[omega21,omega22],
		[omega31,omega32]
		])
	B=np.array([
		[beta1],
		[beta2],
		[beta3]
		])
	X=np.array([[x1],[x2]])
	Y=np.dot(O,X)+B
	return Y

print("individual equations")
print(individual_eq(x1,x2))

print("matrix")
print(matrix_eq(x1,x2))

#===================================================================================================
# Draw the exponential function

# Define an array of x values from -5 to 5 with increments of 0.01
x = np.arange(-5.0,5.0, 0.01)
y = np.exp(x) ;

# Plot this function
fig, ax = plt.subplots()
ax.plot(x,y,'r-')
ax.set_ylim([0,100]);ax.set_xlim([-5,5])
ax.set_xlabel('x'); ax.set_ylabel('exp[x]')
plt.show()
#exp[0]=e^0=1
#exp[1]=e^1~2.72
#exp[-inf]=0
#exp[+inf]= +inf
# the exponential function is convex
#===================================================================================================
# Draw the logarithm function

# Define an array of x values from -5 to 5 with increments of 0.01
x = np.arange(0.01,5.0, 0.01)
y = np.log(x) ;

# Plot this function
fig, ax = plt.subplots()
ax.plot(x,y,'r-')
ax.set_ylim([-5,5]);ax.set_xlim([0,5])
ax.set_xlabel('x'); ax.set_ylabel('$\log[x]$')
plt.show()
#log[0] undefined
#log[1] = 0
#log[e] = 1
#log[e^3] = 3
#log[e^4] = 4
#log[-1] is undefined
#the logarithm function is convave