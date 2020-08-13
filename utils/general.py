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
