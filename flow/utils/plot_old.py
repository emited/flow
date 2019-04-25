from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

import flowlib


def color_code(xlim=(-10, 10), ylim=(-10, 10), res=100):
    x = np.linspace(xlim[0], xlim[1], res)
    y = np.linspace(ylim[0], ylim[1], res)
    X, Y = np.meshgrid(x, y)
    X = np.expand_dims(X, 0)
    Y = np.expand_dims(Y, 0)
    C = np.concatenate([X, Y], axis=0)
    return C
    

def color_code_image(xlim=(-10, 10), ylim=(-10, 10), res=100):
    code = flow_to_image(color_code(xlim, ylim, res))
    return code


def from_matplotlib(fig):
    fig.canvas.draw()
    rgb = fig.canvas.tostring_rgb()
    plt.close(fig)
    data = np.fromstring(rgb, dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_one_image(im, title='', vmin=None, vmax=None):
    if im.shape[0] == 2:
        im = flowlib.flow_to_image(im.transpose(1, 2, 0))
    plt.axis('off')
    plt.imshow(im.squeeze(), origin='lower', vmin=vmin, vmax=vmax)


def plot_images(x, nsample=1):
    """
    x is a list of tuples (name, value) where value
    is a dict of with keys 'in' and 'out', corresponding
    to the input and output sequences given to the model 
    """

    # retrieving max image value to not renorm
    vmin = np.inf
    vmax = - np.inf
    for _, v in x:
        for l in v:
            if v[l][0][0].shape[0] != 2:
                vmin = min(vmin, np.min(v[l]))
                vmax = max(vmax, np.max(v[l]))

    x = OrderedDict(x)

    # calculating max column length
    maxincols, maxoutcols = 0, 0
    for k in x:
        if 'in' in x[k]:
            maxincols = max(len(x[k]['in']), maxincols)
        if 'out' in x[k]:
            maxoutcols = max(len(x[k]['out']), maxoutcols)
    cols = maxincols + maxoutcols
    rows = len(x) * nsample

    res = 1.5
    plt.figure(figsize=(cols * res, rows * res))
    for i, k in enumerate(x):
        for s in range(nsample):
            if 'in' in x[k]:
                title = '{}, in'.format(k)
                for t in range(len(x[k]['in'])):
                    n = s * len(x) * cols + cols * i + t
                    im = x[k]['in'][t][s]
                    plt.subplot(rows, cols, n + 1)
                    plot_one_image(im, title)
            if 'out' in x[k]:
                title = '{}, out'.format(k)
                for t in range(maxincols, maxincols + len(x[k]['out'])):
                    im = x[k]['out'][t - maxincols][s]
                    n = s * len(x) * cols + cols * i + t
                    plt.subplot(rows, cols, n + 1)
                    plot_one_image(im, title)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.001, hspace=0.1)
    return plt.gcf()