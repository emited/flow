from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import cmocean

import torch
import torchvision.utils as vutils

import flowlib


def flow_to_image(flow):
    return flowlib.flow_to_image(flow.transpose(1, 2, 0))


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

def plot_results(x, nsample=1, res=1.5, cmap='viridis'):
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
                vls = [vli[:nsample] for vli in v[l]]
                vmin = min(vmin, np.min(vls))
                vmax = max(vmax, np.max(vls))

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

    def plot_one(im, title='', cmap='viridis'):
        if im.shape[0] == 2:
            im = flow_to_image(im)
        plt.axis('off')
        if hasattr(cmocean.cm, cmap):
            cmap = getattr(cmocean.cm, cmap)
        plt.imshow(im.squeeze(), origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
        #plt.title(title)
    
    plt.figure(figsize=(cols * res, rows * res))
    for i, k in enumerate(x):
        for s in range(nsample):
            if 'in' in x[k]:
                
                title = '{}, in'.format(k)
                for t in range(len(x[k]['in'])):
                    n = s * len(x) * cols + cols * i + t
                    im = x[k]['in'][t][s]
                    plt.subplot(rows, cols, n + 1)
                    plot_one(im, title, cmap=cmap)
            if 'out' in x[k]:
                title = '{}, out'.format(k)
                for t in range(maxincols, maxincols + len(x[k]['out'])):
                    im = x[k]['out'][t - maxincols][s]
                    n = s * len(x) * cols + cols * i + t
                    plt.subplot(rows, cols, n + 1)
                    plot_one(im, title, cmap=cmap, )
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.001, hspace=0.1)
    return plt.gcf()


def plot_tensor(output, x, padding=6, pad_value=1):
    """
    out:model output tensor size TxOxCxHxW
    x: model input arnd target size Tx(I+O)xCxHxW
    """
    samples = []
    for o, xi in zip(output.transpose(0, 1), x.transpose(0, 1)):
        out = vutils.make_grid(o, padding=padding, nrow=1, pad_value=pad_value)
        inn = vutils.make_grid(xi[:-output.size(0)], padding=padding, nrow=1, pad_value=pad_value)
        innout = torch.cat([inn, out], 1)
        targ = vutils.make_grid(xi[-output.size(0):], padding=padding, nrow=1, pad_value=pad_value)
        inntarg = torch.cat([inn, targ], 1)
        sample = torch.cat([inntarg, innout], 2)
        samples.append(sample)
    return torch.cat(samples, 2).transpose(1, 2)


#def plot_color_code_image(xlim=(-10, 10), ylim=(-10, 10), resolution=100, **kwargs):
#    '''For some reason, colors are flipped on the y axis
#    with respect to real middlebury color code.
#    '''
#    plt.title('Middlebury color code')
#    plt.imshow(color_code_image(xlim, ylim, resolution), **kwargs, origin='lower')
#    plt.plot([0, resolution-1], [resolution / 2, resolution / 2], c='black')
#    plt.plot([resolution / 2, resolution / 2], [0, resolution-1], c='black')
#    plt.axis('off')


def plot_flow_quiver(flow, flow_target=None, img=None):
    '''Plots vector field with/without associated image
            according to method given.
            flow: np.ndarray of with size (2, x_dim, y_dim)
            flow_target: np.ndarray of with size (2, x_dim, y_dim)
            img: np.ndarray with size (x_dim, y_dim)

    '''

    if img is not None:
        plt.imshow(img, origin='lower')

    X, Y = np.meshgrid(range(flow.shape[1]), range(flow.shape[2]))

    if flow_target is not None:
        plt.quiver(X[::3, ::3], Y[::3, ::3], flow_target[0, ::3, ::3], flow_target[1, ::3, ::3],
                    color='r', pivot='mid', units='inches')

    plt.quiver(X[::3, ::3], Y[::3, ::3], flow[0, ::3, ::3], flow[1, ::3, ::3],
               pivot='mid', units='inches')


