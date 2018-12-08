import math
import IPython
import itertools
import numpy as np
import matplotlib.pyplot as plt
if IPython.get_ipython(): import ipywidgets as widgets
else: import matplotlib.widgets as widgets
from cycler import cycler
from collections import Iterable
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.figure import figaspect
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection

import feature_visualization as fv
from Kant import ordinal
from switch import switch
from enn_v1_0 import ENN, get_terms_expression
from feature_visualization import md_display


class nn_visualizer(object):
    def __init__(self, x_s, y_s, nn, trace_dict):
        """
        :param x_s:
            2-d array-like. This is supposed to be the numpy-style training \
            set or test set, whereas `x_coords` and `y_coords` below are the \
            corresponding matplotlib-style data!
        """
        """ alternatives: groupby, filter everytime, boolean indexing \
        everytime, np.indices. """
        #         self.x_coords_all_classes, self.y_coords_all_classes =[], []
        #         keyfunc = lambda xy: np.argmax(xy[1], 1)
        #         xy_s = sorted(zip(x_s, y_s), keyfunc)
        #         for k, g in itertools.groupby(xy_s, keyfunc):
        #             """ `g`: xy_class_j """
        #             x_s_class_j = np.array(list(g))[:, 0]
        self.x_s = x_s
        self.y_s = y_s
        """ a indexing array used to select samples of j-th class. """
        self._indices = lambda j: np.where(np.argmax(self.y_s, 1) == j)

        assert hasattr(nn, 'predict') and \
               hasattr(nn, 'logits') and \
               hasattr(nn, 'accuracy') and \
               hasattr(nn, 'config') and \
               hasattr(nn.config, 'num_of_class') and \
               hasattr(nn.config, 'feature_dimension')
#         nn.fp_transformation | nn.feature_transform | nn.
        self.nn = nn

        assert 'W' in trace_dict and \
               'b' in trace_dict
        self.trace_dict = trace_dict

        sampling_magnitude = 10
        self.grid = (sampling_magnitude, sampling_magnitude)
        self.x_s_test = fv.sampling(
            'square', 'numpy', sampling_magnitude)
        
        self.__init__function_sign()
        self.__init__plot_config_setup()
        
    
    """ alternative: design into a function which complete the mapping
    operation every time `functions` parameter is received by `visualize` """
    def __init__function_sign(self):
        self.func_map_dict = {}
        sign = lambda ks, v: self.func_map_dict.update(dict.fromkeys(ks, v))
        sign(
            ['activation_func_plot', 'av_func_plot', 'af_plot', 'afp',
             'av_func', 'af'],
            '_nn_visualizer__activation_func_plot'
        )
        sign(
            ['vars_trace_plot', 'var_trace_plot', 'vt_plot', 'vtp', 
             'vars_plot', 'var_plot', 'trace_plot', 'vars_trace', 'var_trace', 
             'vt', 'vars', 'var'],
            '_nn_visualizer__vars_trace_plot'
        )
        sign(
            ['decision_boundary_plot', 'db_plot', 'decision_boundary', 'db'],
            '_nn_visualizer__decision_boundary_plot'
        )
        sign(
            ['transformation_plot', 'trans_plot','t_plot', 'transformation', 
             'trans', 't'],
            '_nn_visualizer__transformation_plot'
        )
        sign(
            ['image_plot', 'img_plot', 'im_plot', 'imp', 'ip', 'image', 'img', 
             'im', 'i'],
            '_nn_visualizer__image_plot'
        )
        sign(
            ['hist_plot', 'h_plot', 'hp', 'hist', 'h'],
            '_nn_visualizer__hist_plot'
        )
    
    
    def __init__plot_config_setup(self):
        """ initialize the plotting config. """
        fv.style_setup(plt)
        self.plt_cfg = lambda: 'this is structure which holds the plot \
                                config.'
        
        self.plt_cfg.colors = ['b', 'r']
        plt.rc('axes', prop_cycle=cycler(color=self.plt_cfg.colors))
        self.plt_cfg.legend = [Patch(facecolor=c, label='class %s' % (j+1)) 
                                for j, c in enumerate(self.plt_cfg.colors)]

        self.plt_cfg.params_norm = Normalize(
            min(np.min(self.trace_dict['W']), np.min(self.trace_dict['b'])), 
            max(np.max(self.trace_dict['W']), np.max(self.trace_dict['b']))
        )
        """ determine the min and max of logits across the whole training 
        process. """
        min_max_ = [0, 0]
        for params_ in zip(self.trace_dict['W'], self.trace_dict['b']):
            f2_ = self.nn.logits(self.x_s_test, *params_)
            min_, max_ = np.min(f2_.T), np.max(f2_.T)
            if min_max_[0] > min_: min_max_[0] = min_
            if min_max_[1] < max_: min_max_[1] = max_
        self.plt_cfg.logits_norm = Normalize(*min_max_)
        
        self.plt_cfg.af_kwargs = {'cmap': 'coolwarm', 
                                  'norm': self.plt_cfg.logits_norm}
        self.plt_cfg.vt_kwargs = {'type': 'weights', 
                                  'one_off': True,
                                  'cmaps': ['Blues', 'Reds']}
        self.plt_cfg.db_kwargs = {'cmap': 'coolwarm', 
                                  'alpha': 0.7,
                                  'zorder': 5}
        """ `conn_density`: used to denote how many connectionPatch artists to 
        create. """
        self.plt_cfg.trans_kwargs = {'cmap': 'coolwarm', 
                                     'edgecolor': 'white', 
                                     'alpha': 0.5, 
                                     'zorder': -1,
                                     'conn_density': 1,
                                     'conn_colors': 'g',
                                     'conn_alpha': 0.4}
        self.plt_cfg.trans_conn_kwargs = {}
        nn_visualizer.trans_conn_kwargs = {}
        """ `one_off` influence the plot of image-gram of `weights` and 
        `bias`. """
        self.plt_cfg.img_kwargs = {'type': 'data', 
                                   'one_off': True}
        self.plt_cfg.hist_type_arg = 'data'
        self.plt_cfg.hist_kwargs = {'color': 'c', 
                                    'alpha': 0.5, 
                                    'edgecolor': 'k'}
        nn_visualizer.hist_kwargs = self.plt_cfg.hist_kwargs.copy()
        
        """ Attention the `[]` here!!! """
        self.artists_indexing_dict = dict.fromkeys(
            list(self.func_map_dict.values()), None)
        for f_name in self.artists_indexing_dict:
            self.artists_indexing_dict[f_name] = []

        
    """ should I put some responsibilities of argument check into the
    executive function below? """    
    def __plot_config_update(self, *functions, **kwargs):
        """ note that the update of `plt_cfg` is persistent across the whole
        life of one specific `nn_visualizer` object. """
        assert functions, 'No Plotting Functions Given!'
        
        func_to_kwargs_dict = {
            '_nn_visualizer__vars_trace_plot': self.plt_cfg.vt_kwargs,
            '_nn_visualizer__activation_func_plot': self.plt_cfg.af_kwargs,
            '_nn_visualizer__decision_boundary_plot': self.plt_cfg.db_kwargs,
            '_nn_visualizer__transformation_plot': self.plt_cfg.trans_kwargs,
            '_nn_visualizer__image_plot': self.plt_cfg.img_kwargs,
            '_nn_visualizer__hist_plot': self.plt_cfg.hist_kwargs,
        }
        
        def key_map_dict(key):
            for case in switch(key):
                if case('cm'): return 'cmap'; break
                if case('cms'): return 'cmaps'; break
                if case('a'): return 'alpha'; break
                if case('conn') \
                or case('density'): return 'conn_density'; break
                if case('conn_c'): return 'conn_colors'; break
                if case('conn_a'): return 'conn_alpha'; break
                if case('ec'): return 'edgecolor'; break
                if case('c'): return 'color'; break
                if case('default'): return key
        
        """ the children kwargs of them are delivered to the actual function 
        to deal with, instead of `for loop` below (which would need further
        check and matching...no no no!!) """
        for plt_cfg_key, v in kwargs.items():
            """ `_` is a possible f_alias used as prefix. """
            if '_' in plt_cfg_key:
                _, plt_cfg_key = plt_cfg_key.split('_', 1)
                if _ in self.func_map_dict:
                    f_alias = _
                    f_name = self.func_map_dict[f_alias]
                    plt_cfg_key = key_map_dict(plt_cfg_key)
                    func_to_kwargs_dict[f_name][plt_cfg_key] = v
                    continue
                else:
                    plt_cfg_key = _ + '_' + plt_cfg_key

            plt_cfg_key = key_map_dict(plt_cfg_key)
            for f_alias in functions:
                if f_alias in self.func_map_dict:
                    f_name = self.func_map_dict[f_alias]
                    func_to_kwargs_dict[f_name][plt_cfg_key] = v
        
        """ below is the update specific to part of module functions. note
        that `kwargs` below will update as long as the kw is offered
        regardless of whether it will be called. (cuz I don't wanna determine
        if it's in `functions` again ← ←) """
        """ 
        - filter kwargs into `plt_cfg.trans_conn_kwargs` according to prefix
        which is eliminated at the same time (A -> A, B).
        - alternatives to filtering op: filter | map(a.pop, a) | groupby
            - judge n times of A
            - create new (key,value) k(k<n) times into B
            - delete (key,value) k(k<n) times of A
        - ref: dict subtraction: A = A - B <=> B.values = map(A.pop, B)
        """
        _ = [self.plt_cfg.trans_conn_kwargs.update({
            k.replace('conn_', ''): self.plt_cfg.trans_kwargs.pop(k)
        }) for k in self.plt_cfg.trans_kwargs.copy() if 'conn_' in k]
        nn_visualizer.trans_conn_kwargs.update(self.plt_cfg.trans_conn_kwargs)
        
        self.plt_cfg.hist_type_arg = self.plt_cfg.hist_kwargs.pop('type',
                                                                  'data')

        """ update values of `var_trace_plot` kwargs. """
        for case in switch(self.plt_cfg.vt_kwargs['type']):
            if case('all'):    
                assert 'W_grad' in self.trace_dict and \
                       'b_grad' in self.trace_dict and \
                       'loss' in self.trace_dict
                break
            if case('weights') \
                or case('weight') \
                or case('W') \
                or case('w'): 
                self.plt_cfg.vt_kwargs['type'] = 'weights'
                break
            if case('gradients') \
                or case('gradient') \
                or case('grad') \
                or case('g'): 
                assert 'W_grad' in self.trace_dict and \
                       'b_grad' in self.trace_dict
                self.plt_cfg.vt_kwargs['type'] = 'gradients'
                break
            if case('loss') \
                or case('L') \
                or case('l'):
                assert 'loss' in self.trace_dict
                self.plt_cfg.vt_kwargs['type'] = 'loss'
                break
            if case('default'): 
                raise ValueError('Invalid `type` argument of `var_trace_plot`'
                                 ': %s' % self.plt_cfg.vt_kwargs['type'])
    
        """ update values of `image_plot` or `hist_plot` kwargs. """
        def img_hist_type_map_dict(type_):
            for case in switch(type_):
                if case('weights') \
                    or case('bias') \
                    or case('data') \
                    or case('logits'): return type_; break
                if case('weight') \
                    or case('W') \
                    or case('w'): return 'weights'; break
                if case('bia') \
                    or case('b'): return 'bias'; break
                if case('Data') \
                    or case('D') \
                    or case('d'): return 'data'; break
                if case('logit') \
                    or case('outputs') \
                    or case('output') \
                    or case('f'): return 'logits'; break
                if case('default'): 
                    raise ValueError('Invalid `type` argument of `hist_plot`'\
                                     ' or `image_plot`: %s' % type_)
        """ direct assign | setattr | global set | id & set? """
        self.plt_cfg.img_kwargs['type'] = img_hist_type_map_dict(
            self.plt_cfg.img_kwargs['type'])
        self.plt_cfg.hist_type_arg = img_hist_type_map_dict(
            self.plt_cfg.hist_type_arg)
                    
        
    def __is_one_off_mode(self, f_name):
        if (f_name == '_nn_visualizer__vars_trace_plot' and 
            self.plt_cfg.vt_kwargs['one_off']) \
        or (f_name == '_nn_visualizer__image_plot' and 
            self.plt_cfg.img_kwargs['type'] == 'data') \
        or (f_name == '_nn_visualizer__image_plot' and 
            self.plt_cfg.img_kwargs['type'] == 'weights' and
            self.plt_cfg.img_kwargs['one_off']) \
        or (f_name == '_nn_visualizer__image_plot' and 
            self.plt_cfg.img_kwargs['type'] == 'bias' and 
            self.plt_cfg.img_kwargs['one_off']) \
        or (f_name == '_nn_visualizer__hist_plot' and 
            self.plt_cfg.hist_type_arg == 'data'):
            return True
        else:
            return False
    
    
    def __data_and_params_update(self, step):
        self.params = self.trace_dict['W'][step], self.trace_dict['b'][step]

        """ logits of `train` and `test` data. """
        self.f1 = self.nn.logits(self.x_s, *self.params)
        self.f2 = self.nn.logits(self.x_s_test, *self.params)
        self.labels_test = self.nn.predict(
            self.x_s_test, *self.params, one_hot=False
        ).reshape(self.grid)

    
    """ one-off function """
    def dataset_plot(self):
        md_display('Module 1: Visualize the data set')
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        """ alternatives for styling: key-args, decorator, `prop_cycle`. """
        for class_j in range(self.nn.config.num_of_class):
            ax.plot(*self.x_s[self._indices(class_j)].T)
        ax.legend(['class %s' % (j + 1) 
                   for j in range(self.nn.config.num_of_class)])
        plt.show()

    
    """ [one-off | interactive] function  """
    def __vars_trace_plot(self):
        """ var_trace location & switch statements count and location """
        def var_trace_plot(var_trace, var_type):
            assert var_trace.ndim == 3
            
            """ initialize the plotting params. """
            num_of_xcoords = var_trace.shape[0]
            num_of_line = var_trace.shape[1]
            num_of_col = var_trace.shape[2]
            x_coord_1st = 0 if var_type != 'gradients' else 1
            x_coords = range(x_coord_1st, num_of_xcoords + x_coord_1st)
            fig, axes = plt.subplots(
                1, num_of_col, figsize=(8, 4), sharex=True, sharey=True)
            fig.subplots_adjust(bottom=0.13)
#             else:
#             fig, axes = plt.subplots(
#                 2, 1, figsize=(11,22), sharex=True, sharey=True)
            axes = np.array(axes).flatten()

            """ determine the var type and plot. """
            for case in switch(var_type):
                if case('weights') \
                or case('gradients'):
                    legend = [
                        r'$w(%s)$' % term for term in \
                        get_terms_expression(self.nn.fp_transformation)
                    ] + [r'$bias$'] if case('weights') \
                    else [
                        r'$\partial{J}\ / \partial{w_{%sj}}$' % i 
                        for i in range(num_of_line - 1)
                    ] + [r'$\partial{J}\ / \partial{bias_j}$']

                    """ stack `W` and `b` together can avoid redundant \
                    plotting code in the following. """
#                     var_trace =np.vstack((trace_dict['W'], trace_dict['b']))
                    for dim_j in range(num_of_col):
                        for i in range(num_of_line):
                            axes[dim_j].plot(
                                x_coords,
                                var_trace[:, i, dim_j], 
                                linestyle='-', marker=''
                            )
                        """ `legend` should be calculated finally here when \
                        var type is `weights`. """
                        if case('gradients'):
                            axes[dim_j].legend(
                                [_.replace('j', str(dim_j)) for _ in legend])
                        else:
                            axes[dim_j].legend(legend)
                        axes[dim_j].set_title(
                            r'dimension %d of new space' % dim_j, y=1.05)
                        axes[dim_j].set_ylabel(var_type)
                        axes[dim_j].set_xlabel('step')
                        axes[dim_j].set_xticks([
                            _ for _ in x_coords if _ % 2 == 0])
                    plt.show()
                    break
                if case('loss'):
                    """ TODO: loss-element-wise & loss-dim-wise """
                    lc_s, annot_s = [], []
                    for dim_j in range(num_of_col):
                        """
                        fisrt add x coordinates (`stack`), e.g:
                        [0.2    [0.3   ____\   [(0,0.2)    [(0,0.3) 
                         0.6],  -0.4]      /    (1,0.6)],  -(1,0.4)]  
                        then permute dims to adapt to the input format of \
                        `LineCollection` (`transpose`).
                        """
                        lines = np.transpose(
                            np.stack([
                                np.broadcast_to(
                                    x_coords,
                                    (num_of_line, num_of_xcoords)).T,
                                var_trace[:, :, dim_j]
                            ], axis=2),
                            (1, 0, 2)
                        )
                        """ split according to the type of class. """
                        lc_s_dim_j = []
                        cmaps = self.plt_cfg.vt_kwargs['cmaps']
                        assert len(cmaps) == \
                            self.nn.config.num_of_class
                        for cls_j in range(self.nn.config.num_of_class):
                            """ 
                            `ls_cls_j`: lines of class j 
                            `sample_size_j`: sample size of class j
                            """
                            ls_cls_j = lines[self._indices(cls_j)]
                            sample_size_j = len(ls_cls_j)
                            """ attention the two meanings of `j` """
                            lc = LineCollection(
                                ls_cls_j,
                                cmap=cmaps[cls_j]
                            )
                            """ set the line colors """
                            lc.set_array(
                                np.linspace(
                                    int(sample_size_j * 0.2), 
                                    int(sample_size_j * 0.8), 
                                    sample_size_j)
                            )
                            lc.set_clim(0, sample_size_j)
                            axes[dim_j].add_collection(lc)
                            lc_s_dim_j.append(lc)

                            axes[dim_j].plot(
                                x_coords,
                                np.mean(ls_cls_j[:, :, 1], 0),
                                linestyle='-', 
                                color=self.plt_cfg.colors[cls_j], \
                                linewidth=1.5, marker=''
                            )

                        axes[dim_j].autoscale()
                        axes[dim_j].set_title(
                            r'dimension %d of new space' % dim_j, y=1.05)
                        axes[dim_j].set_ylabel('loss per sample')
                        axes[dim_j].set_xlabel('step')
                        axes[dim_j].set_xticks([
                            _ for _ in x_coords if _ % 2 == 0])
                        axes[dim_j].legend(handles=self.plt_cfg.legend)
                        lc_s.append(lc_s_dim_j)

                        """ ref temporarily: https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib """
                        annot = axes[dim_j].annotate(
                            "", xy=(0, 0), xytext=(4, 4),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="c"),
                        )
                        annot.set_visible(True)
                        annot_s.append(annot)

                    def hover(event):
                        if event.inaxes:
                            """ idx of target ax. """
                            idx = list(axes).index(event.inaxes)
                            """ retrieve lcs of all classes in the target ax.\ 
                            """
                            for j, lc in enumerate(lc_s[idx]):
                                is_contain, idx_dict = lc.contains(event)
                                if is_contain:
                                    annot_s[idx].xy = event.xdata, event.ydata
                                    """ ref temporarily: https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement """
                                    """ find the sample size of `j-1` class. \
                                    """
                                    line_quantity_of_pre_classes = \
                                        np.sum(np.argmax(self.y_s, 1) < j) \
                                        if j > 0 else 0
                                    annot_s[idx].set_text(
                                        '%s sample' % \
                                        ordinal(idx_dict['ind'][0] +
                                                line_quantity_of_pre_classes)
                                    )
                                    annot_s[idx].get_bbox_patch(). \
                                        set_alpha(0.4)
                                    annot_s[idx].set_visible(True)
                                    fig.canvas.draw_idle()
                                    return
                        for _ in annot_s: _.set_visible(False)
                        fig.canvas.draw_idle()

                    fig.canvas.mpl_connect("motion_notify_event", hover)
                    plt.show()
                    break
                elif case('default'):
                    raise AssertionError('invalid var type!')
        
        if self.initial:
            md_display('Module 3: Training Process Plotting')
            fv.jtplot.style('chesterish')
            fv.style_setup(plt)
            if self.plt_cfg.vt_kwargs['type'] in ('all', 'weights'):
                var_trace_plot(np.concatenate((
                    self.trace_dict['W'], 
                    self.trace_dict['b'][:,np.newaxis,:]
                ), 1), 'weights')
            if self.plt_cfg.vt_kwargs['type'] in ('all', 'gradients'):
                var_trace_plot(np.concatenate((
                    self.trace_dict['W_grad'], 
                    self.trace_dict['b_grad'][:, np.newaxis, :]
                ), 1), 'gradients')
            if self.plt_cfg.vt_kwargs['type'] in ('all', 'loss'):
                var_trace_plot(self.trace_dict['loss'], 'loss')
            fv.reset()
        else:
            pass

        
    """ obj.__activation_func_plot(){...update...?} """
    """ interactive function  """
    def __activation_func_plot(self):
        def get_algebra_expression(dim_j):
            exp = '+'.join([
                '%.2f' % w + term for w, term in zip(
                    self.params[0][:, dim_j], 
                    get_terms_expression(self.nn.fp_transformation)
                )
            ])
            exp = exp + r'+%.2f' % self.params[1][dim_j]
            exp = exp.replace('+-', '-')
            return r'$f_{dim%s}=%s$' % (dim_j, exp)
        
        if self.nn.config.feature_dimension > 3:
            print('feature dimension of data is greater than 3, only the ' \
                  'algebra expressions are supported.')
            for dim_j in range(len(self.f2.shape[1])):
                md_display(get_algebra_expression(dim_j))
            return
            
        if self.initial:
            md_display('Module 4: Scalar-Valued Function Learned Plotting')

            fig_af = plt.figure(figsize=(8, 4))
            fig_af.subplots_adjust(left=0.05, right=0.89, wspace=0.04)

            axes_af, surf_s = [], []
            num_of_plot = self.f2.shape[1]
            for dim_j in range(num_of_plot):
                ax = fig_af.add_subplot(1, num_of_plot, dim_j+1,
                                        projection='3d')
                ax.set_zlim(self.plt_cfg.logits_norm.vmin,
                            self.plt_cfg.logits_norm.vmax)
                axes_af.append(ax)
            axes_af[0]._shared_x_axes.join(*axes_af)
            axes_af[0]._shared_y_axes.join(*axes_af)

            cbar_ax = fig_af.add_axes([0.92, 0.15, 0.015, 0.7])
            m = cm.ScalarMappable(self.plt_cfg.af_kwargs['norm'],
                                  self.plt_cfg.af_kwargs['cmap'])
            m.set_array([])
            cbar = plt.colorbar(m, cax=cbar_ax)
            self.artists_indexing_dict[
             '_nn_visualizer__activation_func_plot'].append([axes_af, surf_s])
        else:
            axes_af, surf_s = self.artists_indexing_dict[
                '_nn_visualizer__activation_func_plot'][0]
            for _ in surf_s: _.remove()
            surf_s.clear()

        for dim_j, ax in enumerate(axes_af):
            ax.set_title(get_algebra_expression(dim_j), y=1.05)
            surf = ax.plot_surface(
                *self.x_s_test.T.reshape(2, *self.grid),
                self.f2.T[dim_j].reshape(self.grid),
                **self.plt_cfg.af_kwargs
            )
            surf_s.append(surf)

        plt.show()

        
    """ interactive function  """
    """ every func knows only about the message it receives. take func below \
    as an example, so the message is `self.initial`, and it doesn't know the \
    REASON why `self.initial` is that value! """
    def __decision_boundary_plot(self):
        if self.initial:
            md_display('Module 5: Decision Boundary Plotting')

            fig_db, ax_db = plt.subplots(figsize=(4.5, 4.5))
            ax_db.legend(handles=self.plt_cfg.legend)

            for class_j in range(self.nn.config.num_of_class):
                ax_db.plot(*self.x_s[self._indices(class_j)].T, zorder=10)
            self.artists_indexing_dict[
                '_nn_visualizer__decision_boundary_plot'].append([ax_db,'db'])
        else:
            ax_db, db = self.artists_indexing_dict[
                '_nn_visualizer__decision_boundary_plot'][0]
            for _ in db.collections: _.remove()
            acc_train = self.nn.accuracy(self.x_s, self.y_s, *self.params)
            print('Training Accuracy:', acc_train)

        db = ax_db.contourf(
            *self.x_s_test.T.reshape(2, *self.grid),
            self.labels_test,
            **self.plt_cfg.db_kwargs
        )
        self.artists_indexing_dict[
            '_nn_visualizer__decision_boundary_plot'][0][1] = db
        plt.show()
        
    
    """ existence and assignment of kwargs rights and liabilities? """
    @staticmethod    
    def transformation_plot(coords_input, coords_output, labels, title=None, **kwargs):
        """
        :param: coords_input, coords_output
            [ndarray] a ndarray of shape (2, d1, d2, ...). the first two dimension
            corresponds to the x and y coordinates
        :param: labels
            [ndarray] a ndarray of shape (d1, d2, ...). note the di here is the 
            same as above respectively.
        """
        fig_t, axes_t = plt.subplots(1, 2, figsize=(8, 4),)
        fig_t.subplots_adjust(left=0.06, right=0.93, wspace=0.08)
        if not title:
            axes_t[0].set_title('original space', y=1.05)
            axes_t[1].set_title('transformed space', y=1.05)
        else:
            for ax in axes_t: ax.set_title(title, y=1.05)
        
        coords_r = [coords_output, coords_input]
        fv.connection_plot(*reversed(axes_t), 
                           coords_r[0].reshape(2, -1).T, 
                           coords_r[1].reshape(2, -1).T, 
                           colors='coolwarm', alpha=1., **kwargs)
#                          *[coords[i].reshape(2, -1).T for i in [0,1]])
        new_lim = np.min(coords_r), np.max(coords_r)
        new_lim = -3, 3
        for ax, xy_coords_like in zip(reversed(axes_t), coords_r):
            ax.pcolormesh(*xy_coords_like, labels, alpha=0.6, cmap='coolwarm')
            ax.grid(True, linestyle='--', alpha=1.)
            ax.scatter(*xy_coords_like, s=10, c=labels, )
            ax.axis([*new_lim, *new_lim])

#         axes_t[0].dataLim.update_from_data_xy(self.f2, ignore=False)
        plt.show()
        
        
    """ interactive function  """
    def __transformation_plot(self):
        if self.initial:
            md_display('Module 6: Transformation Plotting')

            """ in mpl 2.2.3, there is no need `sharex[y]` any more when \
            `equal` is used. furthermore, the plot will fail to render if 、
            you use them both! """
            trans_s = []
            fig_t, axes_t = plt.subplots(1, 2, figsize=(8, 4),)
            fig_t.subplots_adjust(left=0.06, right=0.93, wspace=0.08)
            """ `axis` will change data lim based current data. """

            for ax in axes_t:
                if ax == axes_t[0]: ax.axis('equal')
                ax.legend(handles=self.plt_cfg.legend)
            axes_t[0].set_title('original space', y=1.05)
            axes_t[1].set_title('transformed space', y=1.05)
            for class_j in range(self.nn.config.num_of_class):
                axes_t[0].plot(
                    *self.x_s[self._indices(class_j)].T, markersize=0.4)
                axes_t[1].plot(*self.f1[self._indices(class_j)].T)
                  
            conn_s = fv.connection_plot(
                axes_t[1], axes_t[0], 
                self.f1, self.x_s, 
                **self.plt_cfg.trans_conn_kwargs,
            )

            """ diagonal line """
            for ax in axes_t:
                ax.plot([-10, 0, 10], [-10, 0, 10], '--k', alpha=0.2, 
                        zorder=-5, linewidth=0.8)
                ax.plot([0], [0], 'ko', markersize=3, zorder=-5)
            """ priority: lim setter > sharex & sharey | autoscale_view"""
            self.artists_indexing_dict[
                '_nn_visualizer__transformation_plot'].\
                append([axes_t, trans_s, conn_s])
        else:
            axes_t, trans_s, conn_s = self.artists_indexing_dict[
                '_nn_visualizer__transformation_plot'][0]
            for _ in trans_s: _.remove()
            trans_s.clear()

            for cls_j in range(self.nn.config.num_of_class):
                axes_t[1].lines[cls_j].set_data(
                    *self.f1[self._indices(cls_j)].T)

            for con, xy in zip(conn_s, self.f1): con.xy1 = xy
        
        for ax, x_s in zip(axes_t, [self.x_s_test, self.f2]):
            trans = ax.pcolormesh(
                *x_s.T.reshape(2, *self.grid),
                self.labels_test,
                **self.plt_cfg.trans_kwargs
            )
            ax.grid(True, linestyle='--', alpha=1.)
            trans_s.append(trans)

        """ css | dpi | figsize, ax size | tight """
        """ set_[]lim, ax.axis, updata_datalim, update_from_..., margin """
        new_xlim = min(-1, self.f2[:, 0].min()), max(1, self.f2[:, 0].max())
        new_ylim = min(-1, self.f2[:, 1].min()), max(1, self.f2[:, 1].max())

        """ ax.update_datalim not work. """
        """ seems the same as `ax.set_[x|y]lim`. """
        for ax in axes_t: ax.axis([*new_xlim, *new_ylim])
#         new_lim = [[min(coords), max(coords)] for coords in self.f2.T]
#         axes_t[1].axis(np.array(new_lim).flatten())

        """ ref: https://stackoverflow.com/questions/7386872/make-matplotlib-autoscaling-ignore-some-of-the-plots """
        """ only `set_lim` and `update_...` together can work? """
        axes_t[0].dataLim.update_from_data_xy(self.f2, ignore=False)
        axes_t[1].dataLim.update_from_data_xy(self.f2, ignore=False)

        plt.show()
        
    
    """ openning-to-outside/static function """
    def image_plot(array_s, title='', norm=None, cols=10):
        assert isinstance(array_s, Iterable), 'no array data found!'
        fv.jtplot.style('gruvboxl')
        rows = math.ceil(len(array_s)/cols)
        fig_im, axes_im = plt.subplots(rows, cols, figsize=(6, 6/(cols/rows)))
        fig_im.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06, )
        fig_im.suptitle(title)

        if not norm:
            norm = Normalize(np.min(array_s), np.max(array_s))
        im_s = []
        for array, ax in zip(array_s, axes_im.flatten()):
            im = ax.imshow(np.atleast_2d(array), cmap='gray', norm=norm)
            im_s.append(im)
        for ax in axes_im.flatten():
            ax.axis('off')
        fv.reset()
        return im_s
        
    
    """ [one-off | interactive] function """
    def __image_plot(self):
        for case in switch(self.plt_cfg.img_kwargs['type']):
            if case('weights') \
            or case('bias'):
                k, idx = ('W', 0) if case('weights') else ('b', 1)
                if self.plt_cfg.img_kwargs['one_off']:
                    """ one-off mode """
                    nn_visualizer.image_plot(
                        self.trace_dict[k], norm=self.plt_cfg.params_norm)
                else:
                    """ interactive mode """
                    if self.initial:
                        im_s = nn_visualizer.image_plot(
                            [self.params[idx]], norm=self.plt_cfg.params_norm)
                        self.artists_indexing_dict[
                            '_nn_visualizer__image_plot'].append([im_s])
                    else:
                        im_s = self.artists_indexing_dict[
                            '_nn_visualizer__image_plot'][0]
                        im_s[0].set_data(np.atleast_2d(self.params[idx]))
                break
            if case('data'):
                """ one-off mode """
                nn_visualizer.image_plot(self.x_s)
                break
            if case('logits'):
                """ interactive mode """
                if self.initial:
                    im_s = nn_visualizer.image_plot(self.f1)
                    self.artists_indexing_dict[
                        '_nn_visualizer__image_plot'].append([im_s])
                else:
                    im_s = self.artists_indexing_dict[
                            '_nn_visualizer__image_plot'][0]
                    for arr, im in zip(self.f1, im_s):
                        im.set_data(np.atleast_2d(arr))

        
    """ openning-to-outside/static function """
    def hist_plot(array_s, suptitle='', bins_num=10, axes_h=None, **kwargs):
        array_s = np.atleast_2d(array_s)
        num_of_hist = len(array_s)
        rows = math.ceil(num_of_hist/3)
        cols = num_of_hist if num_of_hist < 4 else 3
        figsize = (3.5*cols, 3.5*rows) if cols < 3 else (9, 4.5*rows)

        if axes_h is None:
            fig_h, axes_h = plt.subplots(
                rows, cols, figsize=figsize, sharey=True)
            fig_h.subplots_adjust(
                left=0.06, right=0.94, top=0.94, bottom=0.06)
            fig_h.suptitle(suptitle)

        """ supplement the default class-level hist kwargs. """
        nn_visualizer.hist_kwargs.update(kwargs)
        
        min_, max_ = np.min(array_s), np.max(array_s)
        """ amplification factor """
        af = (10**2) if max_-min_ < 5 else 1
        bin_edge_range = [math.floor(min_*af)/af, math.ceil(max_*af)/af]
#         print(bin_edge_range)

        values_s = []
        for arr, ax in zip(array_s, axes_h.flatten()):
            ax.tick_params(labelsize=6.5)
#             import pdb
#             pdb.set_trace()
            values, bin_edge_s, _ = ax.hist(
                arr, 
                np.linspace(*bin_edge_range, bins_num+1), 
                **nn_visualizer.hist_kwargs
            )
            values_s += list(values)
            ax.set_xticks([round(_, 2) for _ in bin_edge_s])
            ax.yaxis.set_tick_params(labelleft=True)
            ax.set_yticks(range(
                0, int(max(values_s)) + 1, math.ceil(max(values_s)/2) ))
        
        plt.show()
        return axes_h

    
    """ [one-off | interactive] function """
    def __hist_plot(self):
        for case in switch(self.plt_cfg.hist_type_arg):
            if case('weights') \
            or case('bias'):
                """ interactive mode """
                params_ = np.row_stack([self.params[0], self.params[1]])
                if self.initial:
                    print('Note: weights and bias are shown together.')
                    axes_h_1 = nn_visualizer.hist_plot(self.params[0], 
                                            'all term features from a dim view')
                    axes_h_2 = nn_visualizer.hist_plot(params_.T, 
                                            'all dims from a term feature view')
                    self.artists_indexing_dict['_nn_visualizer__hist_plot']+=\
                                              [axes_h_1, axes_h_2]
                else:
                    """ `h_s`: all the artists in the hist plot. """
                    axes_h_1, axes_h_2 = self.artists_indexing_dict[
                        '_nn_visualizer__hist_plot']
                    for ax in axes_h_1: ax.cla()
                    for ax in axes_h_2: ax.cla()
                    nn_visualizer.hist_plot(self.params[0], axes_h=axes_h_1)
                    nn_visualizer.hist_plot(params_.T, axes_h=axes_h_2)
                break
            if case('data'):
                """ one-off mode """
#                 axes_h_1 = nn_visualizer.hist_plot([self.x_s], 
#                                         'all dims from a sample view')
                axes_h_2 = nn_visualizer.hist_plot(self.x_s.T, 
                                        'all samples from a dim view')
                self.artists_indexing_dict['_nn_visualizer__hist_plot'] += \
                                          [axes_h_2]
                
                """ x_s_t: transformed feature of `x_s`. """
                x_s_t = self.nn.feature_transform(self.x_s)
#                 axes_h_1 = nn_visualizer.hist_plot(
#                     x_s_t, 'all dims from a transformed sample view')
                axes_h_2 = nn_visualizer.hist_plot(
                    x_s_t.T, 'all transformed samples from a dim view')
                self.artists_indexing_dict['_nn_visualizer__hist_plot'] += \
                                          [axes_h_2]
                break
            if case('logits'):
                """ interactive mode """
                if not self.initial:
                    """ `h_s`: all the artists in the hist plot. """
                    axes_h_2, = self.artists_indexing_dict[
                        '_nn_visualizer__hist_plot']
#                     for ax in axes_h_1: ax.cla()
                    for ax in axes_h_2: ax.cla()
#                     nn_visualizer.hist_plot(self.f1, axes_h=axes_h_1)
                    nn_visualizer.hist_plot(self.f1.T, axes_h=axes_h_2)
                else:
#                     axes_h_1 = nn_visualizer.hist_plot(
#                         self.f1, 'all dims from a logit view')
                    axes_h_2 = nn_visualizer.hist_plot(
                        self.f1.T, 'all logits from a dim view')
                    self.artists_indexing_dict['_nn_visualizer__hist_plot']+=\
                                              [axes_h_2]
                break

                
    def __update(self, step, functions):
        self.initial = False
        self.__data_and_params_update(step)
        for f in functions:
            getattr(self, self.func_map_dict[f])()
        acc_train = self.nn.accuracy(self.x_s, self.y_s, *self.params)
        print('Training Accuracy:', acc_train)

            
    def visualize(self, *functions, **kwargs):
        """
        :params functions:
            a variable length argument, allowing for multiple string
            objects. 
            functions supported: 
            - var_trace_plot
            - activation_function_plot
            - decision_boundary_plot
            - transformation_plot
            - image_plot
            - hist_plot
            
        :params kwargs:
            effective kwargs for `vars_trace_plot`:
            - vars_trace_type: `all` (default) | `weights` | `gradients` | `loss`
            
            effective kwargs for `image_plot`: 
            - image_type: `weights` | `bias` | `data` (deafult) | `logits`; 
            - image_one_off: `True` (default) | `False` (used together with
              `weights` or `bias`);
            
            effective kwargs for `hist_plot`:
            - image_type: `weights` | `bias` | `data` (default) | `logits`;             
        """
        self.initial = True
        self.__plot_config_update(*functions, **kwargs)
        self.__data_and_params_update(step=0)
        
        functions = list(functions)
        for f in functions:
            if f in self.func_map_dict:
                f_name = self.func_map_dict[f]
                getattr(self, f_name)()
                if(self.__is_one_off_mode(f_name)):
                    """ `None` here plays a role of placeholder in order \
                    not to influence the normal iteration of `functions`.\
                    """
                    functions[:0] = [None]
                    functions.remove(f)
            else:
                raise AttributeError('Unknown Plotting Functions!')
        
        """ Attention! any seq used more than once must be list()ed!! """
        functions = list(filter(lambda _: _ is not None, functions))
        if functions:
            widgets.interact(
                self.__update,
                step=widgets.IntSlider(min=0,
                    max=len(self.trace_dict['W'])-1),
                functions=widgets.fixed(functions)
            )