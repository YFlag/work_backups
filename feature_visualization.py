"""
Requirements:
(Writing down makes your thinking clear!)
    - simple plotting:
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
try:
    from jupyterthemes import jtplot
    jtplot.style(theme='chesterish')
    # jtplot.style(context='paper', fscale=0.8, spines=False, gridlines='--')
    # jtplot.reset()
except ImportError:
    print('using default matplotlib style.')
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from switch import switch


""" to be extend: sampling shape, sampling possibly. """
def sampling(sampling_magnitude=5, *, sampling_style='square', \
             data_style='numpy'):
    """
    :param sampling_magnitude:
        int type. this can be viewed as 'an adaptive sample size', which is \
        generated according to the `sampling_style` and feature dimension.
    """
    if data_style != 'numpy':
        print('please use numpy-style currently.')
        return
    for case in switch(sampling_style):
        if case('quadratic'):
            x = np.linspace(-1, 1, sampling_magnitude)
            result = np.column_stack([
                x,
                0.5 * np.sin(3.1*(x-0.5))
            ])
            break
        if case('square'):
            result = np.column_stack([
                np.repeat(np.linspace(-1, 1, sampling_magnitude), \
                          sampling_magnitude),
                np.tile(np.linspace(-1, 1, sampling_magnitude), \
                        sampling_magnitude),
            ])
            break
        if case('line'):
            result = np.column_stack([
                np.zeros(sampling_magnitude),
                np.linspace(-1, 1, sampling_magnitude)
            ])
        if case('random'):
            result = np.random.random((sampling_magnitude, 2))
            break
        if case('default'):
            raise AssertionError('illegal sampling style!')
    for case in switch(data_style):
        if case('numpy'): return result; break
        if case('matplotlib'): return result.T; break
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


""" transformation visualization interface. """
""" high cohesion、low coupling!!! Remember this!! """
def feature_plot(
    *xy_coords, labels=None, label_method=None, connection_density=0, \
    fig_size=(6, 6), tick_labelsize=9, spine_visible=False, line_style='co', \
    markersize=3, connection_color='g'
    ):
    """
    :param xy_coords:
        array-like type. the data structure is supposed to hold a matplotlib- \
        style, i.e. `xy_coords[0]` corresponds to x and y coordinates for the \
        first graph, `xy_coords[1]` corresponds to x and y coordinates for the \
        second graph, and so on.
    """

    assert xy_coords != ()
    xy_coords = np.array(xy_coords)
    quantity_of_graph = len(xy_coords)
    sample_size = xy_coords[0, 0].size
    """ the inputs (X, Y, C) of `pcolormesh` must be a 2-D array. """
    if label_method: 
        assert label_method in ('pcolormesh', 'tripcolor', 'contour', \
            'contourf', 'tricontour', 'tricontourf') and labels is not None
    
    fig, axes = plt.subplots(
        nrows=1, ncols=quantity_of_graph,
        figsize=(fig_size[0] * quantity_of_graph, fig_size[1]),
        sharex=True, sharey=True
    )
    axes = np.array(axes).flatten()

    """ a boolean value indicating triple """
    FO_L_LM = np.array(
        [quantity_of_graph == 2, labels is not None, bool(label_method)],
    ).astype(np.int)
    for i, ax in enumerate(axes):
        ax.tick_params(labelsize=tick_labelsize)
        for spine in ax.spines.values(): spine.set_visible(spine_visible)
        """ see `switch.py` for more details. """
        for case in switch(FO_L_LM, np.array_equal):
            if case([0, 0, 0]) \
            or case([1, 0, 0]):
                ax.plot(*xy_coords[i], line_style, markersize=markersize)
                break
            if case([0, 1, 0]) \
            or case([1, 1, 0]):
                ax.scatter(*xy_coords[i], c=labels)
                break
            if case([1, 1, 1]):
                ax.plot(*xy_coords[i], line_style, markersize=markersize)
                getattr(ax, label_method)(*xy_coords[i], labels)
                # cmap='RdBu_r', #facecolor='none', edgecolor='k', alpha=0.1
                break
            if case([0, 1, 1]):
                getattr(ax, label_method)(*xy_coords[i], labels)
                break
            if case('default'):
                raise AssertionError('Argument Error!')
                
    if connection_density:
        """ 
        `i`: sample index 
        `sampling size`: sampling number in all samples
        """
        assert quantity_of_graph == 2
        xy_coords = xy_coords.reshape(2, 2, -1)
        for i in np.linspace(
                0, sample_size - 1,
                round(sample_size * connection_density)
        ).astype(np.int):
            """ 0 or 1-th phase | x or y-coordinates | i-th sample """
            """ `axesB` is the first one. """
            """ 
            an equivalent form to params `xyA` and `xyB` assignment:
            *xy_coords[: : -1, :, i],
            """
            con = ConnectionPatch(
                xyA=xy_coords[1][:, i], xyB=xy_coords[0][:, i],
                coordsA = 'data', coordsB = 'data',
                arrowstyle = '<-',
                axesA = axes[1], axesB = axes[0],
                linewidth = 1.,
                color = connection_color
            )
            axes[1].add_artist(con)
    
    plt.axis('equal')
    plt.show()


"""
lesson:
    - clear your boundary of interface.
    - attention the multi-state combination occasion.
"""