import matplotlib.pyplot as plt
import numpy as np


def split_box(ax, n, x0, y0, x1, y1, min_split_frac=0.2, min_split_size=80, change_x=True):
    clr_str = 'bgrcmyk'
    b = []
    if n >= 0:
        b = [[x0, y0, x1, y1]]
        r = np.random.random() * (1 - 2 * min_split_frac) + min_split_frac
        if change_x and np.abs(x1 - x0) >= min_split_size:
            dx = (x1 - x0) * r
            b = b + split_box(ax, n - 1, x0, y0, x0 + dx, y1, change_x=False, min_split_size=min_split_size)
            b = b + split_box(ax, n - 1, x0 + dx, y0, x1, y1, change_x=False, min_split_size=min_split_size)
        if (not change_x) and np.abs(y1 - y0) >= min_split_size:
            dy = (y1 - y0) * r
            b = b + split_box(ax, n - 1, x0, y0, x1, y0 + dy, min_split_size=min_split_size)
            b = b + split_box(ax, n - 1, x0, y0 + dy, x1, y1, min_split_size=min_split_size)
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], linewidth=0.8, color=clr_str[n % len(clr_str)])
    return b


if __name__ == "__main__":
    plt.close("all")
    for n in range(1, 6):
        nrows = 4
        fig, ax = plt.subplots(nrows=nrows, figsize=(4, 8))
        fig.suptitle('n = ' + str(n), fontsize=16)
        plt.tight_layout()
        for i in range(nrows):
            b = split_box(ax[i], n, 0, 0, 2000, 1000, min_split_size=300)
        plt.show()
