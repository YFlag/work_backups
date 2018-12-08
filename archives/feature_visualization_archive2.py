"""
the original version of `feature_plot` method.
"""

from itertools import zip_longest
from matplotlib.ticker import FormatStrFormatter


""" transformation visualization interface. """
""" high cohesion、low coupling!!! Remember this!! """
def feature_plot(
    *xy_coords, labels=None, label_method=None, connection_density=0, \
    fig_size=(6, 6), tick_labelsize=9, spine_visible=False, line_styles='c', \
    markersize=3, connection_color='g', legend=None, cmap=None, x_discrete=False, \
    x_highlight=-1, axis_title=None, axis_label=None
    ):
    """
    :param xy_coords:
        array-like type. the data structure is supposed to hold a matplotlib- \
        style, i.e. `xy_coords[0]` corresponds to x and y coordinates for the \
        first graph, `xy_coords[1]` corresponds to x and y coordinates for the \
        second graph, and so on...
        for the shape of any `xy_coords[i]`, it could be (2, ...), whose `2` \
        corresponds to x-coord set and y-cood set, and could also be (4, ...), \
        corresponds to x, y, x, y, and so on...
    """

    xy_coords = np.array(xy_coords)
#     print(xy_coords)
#     print(xy_coords.shape)
    assert xy_coords.size != 0 and len(xy_coords.shape) >= 3 and xy_coords.shape[1] % 2 == 0
    
    line_styles = line_styles.split('|')
    quantity_of_graph = len(xy_coords)
    sample_size = xy_coords[0, 0].size
    """ the inputs (X, Y, C) of `pcolormesh` must be a 2-D array. """
    if label_method: 
        assert label_method in ('pcolormesh', 'tripcolor', 'contour', \
            'contourf', 'tricontour', 'tricontourf') and labels is not None
    
    fig, axes = plt.subplots(
        nrows = int((quantity_of_graph - 1) / 3) + 1, 
        ncols = quantity_of_graph if quantity_of_graph <=3 else 3,
        figsize = (fig_size[0] * quantity_of_graph, fig_size[1]),
        sharex = True, sharey = True
    )
    axes = np.array(axes).flatten()

    """ a boolean value indicating triple. `MG`: multiple graphs. """
    MG_L_LM = np.array(
        [quantity_of_graph != 1, labels is not None, bool(label_method)],
    ).astype(np.int)
    for idx, ax in enumerate(axes):
        ax.tick_params(labelsize=tick_labelsize)
        if axis_title: 
            axis_title_ = axis_title
            ax.set_title(axis_title_.replace('j', str(idx)), y=1.05)
        if axis_label:
            ax.set_xlabel(axis_label[0], labelpad=15)
            if len(axis_label) > 1: ax.set_ylabel(axis_label[1])
        for spine in ax.spines.values(): 
            spine.set_visible(spine_visible)
        
        """ see `switch.py` for more details. """
        for case in switch(MG_L_LM, np.array_equal):
            if case([0, 0, 0]) \
            or case([1, 0, 0]):
                """ simple plotting & multiple simple plotting """
                """ e.g. (2,n) / (4,n,n) -> (1,2,n) / (2,2,n,n) """
                xy_coords_idx = xy_coords[idx].reshape(-1, 2, *xy_coords[idx].shape[1:])
#                 print(xy_coords_i)
                """ if no enough styles, use `` for default. """
                for xy, line_style in zip_longest(xy_coords_idx, line_styles, fillvalue=''):
                    ax.plot(*xy, line_style, markersize=markersize)
#                     ax.grid(True, linestyle='--', alpha=1.)
#                 for line, line_style in zip(lines, line_styles):
#                     line.set_linestyle(line_style)
                if legend: 
                    legend_ = legend.copy()
                    for j in range(len(legend_)): 
                        legend_[j] = legend_[j].replace('j', str(idx))
                    ax.legend(legend_)
                if x_discrete:
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                if x_highlight != -1 and x_highlight >= np.min(xy_coords_idx[:, 0, ...]):
                    ax.axvspan(x_highlight-0.1, x_highlight+0.1, facecolor='w')
                break
            if case([0, 1, 0]) \
            or case([1, 1, 0]):
                """ ditto """
                ax.scatter(*xy_coords[idx], c=labels)
                break
            if case([1, 1, 1]):
                """ transformation plotting """
                ax.axis('equal')
                ax.plot(*xy_coords[idx], line_styles[0], markersize=markersize)
                getattr(ax, label_method)(*xy_coords[idx], labels)
                # cmap='RdBu_r', #facecolor='none', edgecolor='k', alpha=0.1
                break
            if case([0, 1, 1]):
                """ decision boundary / region plotting """
                ax.grid(linestyle='--', alpha=0.5)
                getattr(ax, label_method)(*xy_coords[idx], labels, cmap=cmap)
                ax.legend(handles=[
                    Patch(facecolor='r', label='class 1'),
                    Patch(facecolor='b', label='class 2')
                ])
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
    
#     if quantity_of_graph == 2: plt.axis('equal')
    plt.show()
