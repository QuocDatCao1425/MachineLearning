import numpy as np
import matplotlib.pyplot as plt

# Define the loss function
def loss_function(phi):
    return 1 - 0.5 * np.exp(-(phi - 0.65)**2 / 0.1) - 0.45 * np.exp(-(phi - 0.35)**2 / 0.02)

# Plotting function
def draw_function(loss_function, a=None, b=None, c=None, d=None):
    phi_plot = np.arange(0, 1, 0.01)
    fig, ax = plt.subplots()
    ax.plot(phi_plot, loss_function(phi_plot), 'r-')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$L[\phi]$')
    if a is not None and b is not None and c is not None and d is not None:
        plt.axvspan(a, d, facecolor='k', alpha=0.2)
        ax.plot([a, a], [0, 1], 'b-')
        ax.plot([b, b], [0, 1], 'b-')
        ax.plot([c, c], [0, 1], 'b-')
        ax.plot([d, d], [0, 1], 'b-')
    plt.show()

# Line search algorithm
def line_search(loss_function, thresh=0.0001, max_iter=10, draw_flag=False):
    a = 0
    b = 0.33
    c = 0.66
    d = 1.0
    n_iter = 0

    while np.abs(b - c) > thresh and n_iter < max_iter:
        n_iter += 1
        lossa = loss_function(a)
        lossb = loss_function(b)
        lossc = loss_function(c)
        lossd = loss_function(d)

        if draw_flag:
            draw_function(loss_function, a, b, c, d)

        print(f'Iter {n_iter}, a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}')

        # Rule #1: If lossa is less than lossb, lossc, and lossd, shrink interval around a
        if lossa < lossb and lossa < lossc and lossa < lossd:
            b = (a + b) / 2
            c = (a + c) / 2
            d = (a + d) / 2
            continue

        # Rule #2: If lossb < lossc, adjust interval so that d becomes c
        if lossb < lossc:
            d = c
            b = a + (d - a) / 3
            c = a + 2 * (d - a) / 3
            continue

        # Rule #3: If lossc < lossb, adjust interval so that a becomes b
        if lossc < lossb:
            a = b
            b = a + (d - a) / 3
            c = a + 2 * (d - a) / 3
            continue

    # The final solution is the average of b and c
    soln = (b + c) / 2
    return soln

# Run the line search algorithm
soln = line_search(loss_function, draw_flag=True)
print(f'Soln = {soln:.3f}, loss = {loss_function(soln):.3f}')

