import numpy as np


def printProgressBar(text, current_value, max_value, writeover=True):
    start = '\r' if writeover else ' '
    print("{start}{text}: {prog:>5}%".format(start=start, text=text, prog=round(100 * current_value / max_value, 3)), end='')

def pickRandomColor(threshold=0.01, individual=False):
    # initalized to black if threshold=0
    color = np.zeros(3)
    val = np.Inf
    if individual:
        while val < threshold or val > 1-threshold:
            color = np.random.rand(3)
            val = max(color)
    else:
        while val < threshold or val > 3-threshold:
            color = np.random.rand(3)
            val = sum(color)
    return color

def print_eqns_Ab(rectangle):
    """To see paste into https://www.desmos.com/calculator
       rounding as desmos interprets 'e' in terms of exponential
       not scientific notation"""
    A = np.around(rectangle.A, 2)
    b = np.around(rectangle.b, 2)
    for i in range(np.shape(rectangle.A)[0]):
        print(f'{A[i][0]}x + {A[i][1]}y <= {b[i][0]}')


def print_eqns_Ac(rectangle):
    """"Verification, should print same as above, disregarding rounding error"""
    A = np.around(rectangle.A, 2)
    c = np.around(rectangle.c, 2)
    for i in range(np.shape(rectangle.A)[0]):
        print(f'{A[i][0]}(x - {c[i][0][0]}) + {A[i][1]}(y - {c[i][1][0]}) <= 0')
