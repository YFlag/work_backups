# modules to import
%matplotlib inline
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import ConnectionPatch

"""
transformation visualization interface.
"""
def transform_visualization(feature_input_s, transform_matrix = None, labels = None, connection = False, *, 
        fig_size = (6, 6), 
        tick_labelsize = 9,
        spine_visible = False,
        markersize = 3
    ):
    """
    :param feature_input_s:
        any 2d sequence type. the data structure is supposed to hold a numpy-style, i.e. row # corresponds \
        to sample size and col # corresponds to feature dimension.
    :param transform_matrix:
        any 2d sequence type. Note that the element of this param can be a function, which is acted on the \
        `feature_input` as a whole (actually this is so-called vector-valued function in this case, which can be viewed as a \
        general form of parametric equations). Only visualize the `feature_input_s` if None.
    """
    feature_input_s = np.array(feature_input_s)
    # `sampling_size` should be a number which can be rooting to a natural number.
    sampling_size = len(feature_input_s)
    
    # 'callable()' returns True denotes the element data type is a function object here.
    if transform_matrix and callable(transform_matrix[0][0]):
        vector_valued_function = np.array(transform_matrix).reshape(-1)        
        feature_output_s_collector = []
        
        # here, `feature_function` receives a vector and gives a scalar.
        for feature_input in feature_input_s:
            feature_output_s_collector.append([
                feature_function(*feature_input) for feature_function in vector_valued_function
            ])
        feature_output_s = [np.array(feature_output_s_collector)]
    elif transform_matrix:
        feature_output_s = [feature_input_s.dot(transform_matrix)]
    else:
        feature_output_s = []
        
# print(feature_output_s)
            
    # plotting
    def feature_format_conversion(feature_s_all_phase):
        # set lim range here!
        nonlocal lim_x, lim_y
        
        if feature_s_all_phase.shape[1] == 1:
            x_coordinates = np.array([0] * feature_s_all_phase.shape[0])
        elif feature_s_all_phase.shape[1] == 2:
            x_coordinates = feature_s_all_phase[:, 0]
            if lim_x[0] is None or lim_x[0] > min(x_coordinates): lim_x[0] = min(x_coordinates)                
            if lim_x[1] is None or lim_x[1] < max(x_coordinates): lim_x[1] = max(x_coordinates)
        y_coordinates = feature_s_all_phase[:, -1]
        if lim_y[0] is None or lim_y[0] > min(y_coordinates): lim_y[0] = min(y_coordinates)                
        if lim_y[1] is None or lim_y[1] < max(y_coordinates): lim_y[1] = max(y_coordinates)
        if sampling_size_per_dimension.is_integer():
            _ = int(sampling_size_per_dimension), int(sampling_size_per_dimension)
            return x_coordinates, y_coordinates
# return x_coordinates.reshape(_), y_coordinates.reshape(_)
        else:
            return x_coordinates, y_coordinates
    
    quantity_of_col = 2 if transform_matrix else 1
    fig, axes = plt.subplots(
        nrows = 1, ncols = quantity_of_col,
        figsize = (fig_size[0]*quantity_of_col, fig_size[1])
    )
    
    feature_s_all_phase = [feature_input_s] + feature_output_s
    axes = np.array(axes).reshape(-1)
    sampling_size_per_dimension = sampling_size ** (1/2)
    lim_x, lim_y = [[None, None]] * 2
    coordinates_all_phase = [feature_format_conversion(_) for _ in feature_s_all_phase]
    
    for i, ax in enumerate(axes):
        ax.tick_params(labelsize=tick_labelsize)
        for spine in ax.spines.values(): spine.set_visible(spine_visible)
        if labels is None:
            artist = ax.plot(*coordinates_all_phase[i], 'yo', markersize=markersize, zorder=5)
        elif (labels is not None) and sampling_size_per_dimension.is_integer():
            # the inputs (X, Y, C) of `pcolormesh` must be a 2-D array.
            # alternatives to `pcolormesh`: using plot twice with their directions mutually perpendicular.
            artist = ax.tripcolor(
                *coordinates_all_phase[1],
                labels,
# labels.reshape(int(sampling_size_per_dimension), int(sampling_size_per_dimension)),
                #cmap='RdBu_r', #facecolor='none', edgecolor='k', alpha=0.1 
            )
    
    if connection:
        coordinates_all_phase_ = np.array(coordinates_all_phase).reshape(2, 2, -1)
        for i in range(sampling_size):
            if i % 1 == 0:
                # 0-th phase | x-coordinates | i-th sample
                pointA = coordinates_all_phase_[0][0][i], coordinates_all_phase_[0][1][i]
                pointB = coordinates_all_phase_[1][0][i], coordinates_all_phase_[1][1][i]
        # print(pointA, pointB)
                # `axesB` is the first one.
                con = ConnectionPatch(xyA=pointB, xyB=pointA, 
                                  coordsA='data', coordsB='data', 
                                  arrowstyle='<-',
                                  axesA=axes[1], axesB=axes[0],
                                  linewidth=0.5,
                                  color='r'
                )

    # using `plt.xlim` will cause some weird problems in matplotlib-2.x.x.
    for ax in axes: ax.set(xlim=lim_x, ylim=lim_y)
    plt.show()