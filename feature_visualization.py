"""
Requirements:
(Writing down makes your thinking clear!)
    - simple plotting
    - transformation plotting
    - decision region plotting
    - decision boundary plotting
    - continuous version of 1-4

Methods of plotting:
    - plot, scatter
    - contour, contourf, tricontour, tricontourf
    - pcolormesh, tripcolor

Matching:
    - simple... <-> plot/scatter
    - transfor... <-> plot/scatter [patch]
    - region/boundary... <-> patch

Design:
    - data preparation
    - axes layout preparation
    - plotting
================================================================================
"""


""" modules to import """
import matplotlib as mpl
rcParamsInline = dict(mpl.rcParams)
try:
    from jupyterthemes import jtplot
#     jtplot.style(theme='gruvboxl')
    # jtplot.style(context='paper', fscale=0.8, spines=False, gridlines='--')
except ImportError:
    print('using default matplotlib style.')
import IPython
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from matplotlib.patches import Patch, ConnectionPatch

from switch import switch


""" to be extend: sampling shape, sampling possibly. """
def sampling(sampling_style, data_style='numpy', sampling_magnitude=5):
    """
    :param sampling_magnitude:
        int type. this can be viewed as 'an adaptive sample size', which is \
        generated according to the `sampling_style` and feature dimension.
    """
    for case in switch(sampling_style):
#         print(sampling_style)
        """ do remember write the `break`!!!!!!! """
        if case('quadratic'):
            x = np.linspace(-1, 1, sampling_magnitude)
            result = np.column_stack([
                x,
                0.5 * np.sin(3.1*(x-0.5))
            ])
            break
        if case('square'):
#             if data_style == 'numpy':
            result = np.column_stack([
                np.repeat(np.linspace(-1, 1, sampling_magnitude), \
                          sampling_magnitude),
                np.tile(np.linspace(-1, 1, sampling_magnitude), \
                        sampling_magnitude),
            ])
#                 .reshape(sampling_magnitude, sampling_magnitude, 2)
#             elif data_style == 'matplotlib':
#                 return np.mgrid[-1:1:2/sampling_magnitude, -1:1:2/sampling_magnitude]
#             else:
#                 raise AssertionError('illegal data style!')
            break
        if case('line'):
            result = np.column_stack([
                np.zeros(sampling_magnitude),
                np.linspace(-1, 1, sampling_magnitude)
            ])
            break
        if case('random'):
            result = np.random.random((sampling_magnitude, 2))
            break
        if case('default'):
            raise AssertionError('illegal sampling style!')
    for case in switch(data_style):
        if case('numpy'): return result; break
        if case('matplotlib'): return result.T; break
#                 return np.stack((result[..., 0], result[..., 1]))
        if case('default'): raise AssertionError('illegal data style!')


def transform(feature_input_s, transform_matrix):
    """
    :param feature_input_s:
        any 2d array-like type. This is supposed to hold a numpy-style data \
        structure. i.e. row # corresponds to sample size and col # corresponds \
        to feature dimension.
    :param transform_matrix:
        any 2d array-like type. Note that the element of this param can be a \
        function, which is acted on the `feature_input` as a whole (actually \
        this is so-called vector-valued function in this case, which can be \
        viewed as a general form of parametric equations).
    """
    assert not transform_matrix or np.array(transform_matrix).ndim == 2
    """ `callable()` returns True denotes the element data type is a \
    function object here. """
    if transform_matrix and callable(transform_matrix[0][0]):
        vector_valued_function = np.array(transform_matrix).reshape(-1)
        feature_output_s_collector = []

        """ here, `feature_function` receives a vector and gives a scalar. \
        """
        for feature_input in feature_input_s:
            feature_output_s_collector.append([
                feature_function(*feature_input) for feature_function in \
                vector_valued_function
            ])
        return np.array(feature_output_s_collector)
    elif transform_matrix:
        return feature_input_s.dot(transform_matrix)


def reset():
    """ ref: https://github.com/dunovank/jupyter-themes/blob/master/jupyterthemes/jtplot.py """
    colors = [(0., 0., 1.), (0., .5, 0.), (1., 0., 0.), (.75, .75, 0.),
            (.75, .75, 0.), (0., .75, .75), (0., 0., 0.)]
    for code, color in zip("bgrmyck", colors):
        rgb = mpl.colors.colorConverter.to_rgb(color)
        mpl.colors.colorConverter.colors[code] = rgb
        mpl.colors.colorConverter.cache[code] = rgb
    mpl.rcParams.update(rcParamsInline)
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['axes.facecolor'] = 'white'
#     mpl.rcParams['figure.dpi'] = 72
    
    
def style_initialize_(axes):
    axes = np.array(axes).flatten()
    for ax in axes:
        ax.grid(linestyle='--', alpha=1.)
        ax.tick_params(labelsize=9)
        for spine in ax.spines.values(): spine.set_visible(False)


def style_initialize(plt):
    plt.rc(
        'axes', 
        grid=True,
        titlesize=10,
        labelsize=10,
        labelpad=7
    )
    plt.rc(
        'axes.spines',
        left=False,
        bottom=False,
        top=False,
        right=False,        
    )
    plt.rc(
        'font',
        size=9
    )
    plt.rc(
        'lines', 
        marker='o', 
        markersize=1,
        linestyle=' ',
    )
    plt.rc('grid', linestyle='--')
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7)
    plt.rc('legend', fontsize=8)
    

template = """<font face='Goudy Old Style' size=5><span style="color:rgb(0, 92, 84);">%s</span></font>"""
template_2 = """<font face='Goudy Old Style'>%s</font>"""

def md_display(text):
    if IPython.get_ipython():
        display(Markdown(template % text))
    else:
        print(text)
    

def connection_plot(axA, axB, dataA, dataB, density=1, *, color='g'):
    """ 
    `i`: sample index 
    `sampling size`: sampling number in all samples
    """
    assert dataA.shape == dataB.shape and dataA.ndim == 2
    sample_size = dataA.shape[0]
    con_s = []
    for i in np.linspace(
        0, sample_size - 1, round(sample_size * density)
    ).astype(np.int):
        """ `axesB` is the first one. """
        """ 
        an equivalent form to params `xyA` and `xyB` assignment:
        *xy_coords[: : -1, :, i],
        """
        con = ConnectionPatch(
            xyA=dataA[i], xyB=dataB[i],
            coordsA='data', coordsB='data',
            arrowstyle='<-',
            axesA=axA, axesB=axB,
            linewidth=0.7, color=color, alpha=0.5,
        )
        axA.add_artist(con)
        con_s.append(con)
    return con_s


"""
lesson:
    - clear your boundary of interface: high cohesion、low coupling!!! Remember this!!
    - attention the multi-state combination occasion.
"""