""""
Design of the trail:
1.Structure
- dimension mapping in the whole space transform process: (#1, #2, \
  [...]). each # corresponds to an independent space, whose dimension is \
  the value for the #.
- design of non-linear scalar-valued function of each dimension above: \
  polynomial functions.
2.Fitting Algorithm
- gradient descent
"""

import random
import itertools
import numpy as np

from switch import switch

"""
We start by designing the polynomial function of degree 3. To consider all possible terms \
in polynomials, we can leverage the method of Cartesian product and make some changes on \
it.
"""

"""
1. generate final result directly by something like `np.identity`?
2. ideas about store type: 
    - pure ndarray implementation (order is clear).
    - set(term_collection) -> set -> list.
3. structure design of `term`:
    - value('x_1'), power | matching by index -> not clear even possible.
    - 'x_1', power | matching by str -> more clear, and matching by str is easy.
    - [power1, power2, ...] for each idx -> less clear meaning, more memory, but more clear in structure. 
"""
def get_terms_collection(quantity_of_vars, max_df):
    """ 
    :param max_df:
        - [int] degree of freedom
    :return: 
        - [list] a list of which each element is 2d ndarray type. Note the returned type is
          LIST instead of ndarray!
    """
    if max_df == 1:
        return [np.identity(quantity_of_vars, dtype=np.int32)]

    """
    1. another idea for implementing `terms_collection_s`: add a new arg \
    'all_pre_terms_collection' in function.
    """
    """ 1~n-1 | 1~n """
    one_to_max_minus_one_df_collection = get_terms_collection(
        quantity_of_vars, max_df-1)
    one_to_max_df_collection = one_to_max_minus_one_df_collection

    """ n-1 | n """
    max_minus_one_df_collection = one_to_max_minus_one_df_collection[-1]
    max_df_collection = []

    for idx in range(quantity_of_vars):
        for pre_term in max_minus_one_df_collection:
            pre_term_ = pre_term.copy()
            pre_term_[idx] += 1
            max_df_collection.append(pre_term_)
        """
        another idea: generate `term_collector` by filtering something like \
        [1, 2], [2, 1] by set? seems wrong.
        """
        max_minus_one_df_collection = list(filter(
            lambda term: term[idx] == 0,
            max_minus_one_df_collection
        ))
    one_to_max_df_collection.append(
        np.array(max_df_collection))
    
    return one_to_max_df_collection


def get_terms_expression(terms_collection):
    terms_expression = []
    for term_i in np.vstack(terms_collection):
        term_i_exp = ''.join([
                 r'' if not pow_ 
            else r'x_%s' % k if pow_ == 1 
            else r'x^%s_%s' % (pow_, k)
            for k, pow_ in enumerate(term_i)
        ])
        terms_expression.append(term_i_exp)
    return terms_expression


def feature_polynomial_transform(feature_input_s, terms_collection):
    """ x^T -> T(x^T) """
#     import pdb
#     pdb.set_trace()
    return feature_input_s
#     fp_transformation = np.vstack(terms_collection)
#     feature_output_s = []
#     for x in np.atleast_2d(feature_input_s):
#         feature_output_s.append([
#             sum(var ** power if power != 0 else 0 for var, power in zip(
#                 x, term)) for term in fp_transformation
#         ])            
#     return np.array(feature_output_s)


def get_non_linear_transform(T, terms_collection, space_dimension=2):
    """ non-linearize the linear transformation `T`. returns a function. """
    terms_collection = np.vstack(terms_collection)
    assert terms_collection.ndim == space_dimension
    return lambda feature_input_s: feature_polynomial_transform(feature_input_s, 
                                                                terms_collection) @ T


def non_linear_transform(input_s, feature_candidates, *linear_transform):
    """ 
    Non-linearize all linear transformations in `linear_transform`, and
        stack them together to get a mixed-style non-linear transformation.
        then apply it to the inputs. note that the non-linear map obtained is 
        not unique and are varied according to `feature_candidates`. all 
        possible output_s are packed into a list `all_feature_s_output_s`.
    
    :param feature_candidates:
        [ndarray] a sequence of polynomial terms, each of which is represented 
        by a 1d array in the sequence. e.g. `[1, 0, 2]` stands for `x * z^2`.
        note that the number of terms should be enough for the non-linear
        transformation to sample. Precisely, it should be larger than sum of
        row # of all linear transformations in `linear_transform`.
    :param linear_transform:
        [variable-length arguement] each element is a valid 2d array-like (matrix)
        which denotes a linear transformation.
    """
    input_s = np.atleast_2d(input_s)

    feature_candidates = np.row_stack(feature_candidates)
    assert input_s.ndim == 2 and \
           feature_candidates.ndim == 2 and \
           input_s.shape[1] == feature_candidates.shape[1] and \
           linear_transform
    linear_transform = np.row_stack(linear_transform)
    
    sample_size = input_s.shape[0]
    original_space_dimension = input_s.shape[1]
    latent_space_dimension = linear_transform.shape[0]
    
    all_feature_s_output_s = []
    for feature_s in itertools.combinations_with_replacement(
        feature_candidates, latent_space_dimension):
        all_feature_s_output_s.append((
            feature_s, 
            feature_polynomial_transform(input_s, feature_s) @ \
            linear_transform
        ))
    return all_feature_s_output_s


class ENN():
    def __init__(self, **config):
        """ 
        initialize the default config. ref: see `simple_nn.ipynb`.
        
        :param config:
            - max_degree: 1 (default)
            - feature_dimension: 2 (default)
            - num_of_class: 2 (default) [design into dynamic and determined in train?]
            - space_mapping_process
            - params_init: `normal` (default) | `random` | `identity`
            - learning_rate: 1 (default)
            - batch_size: 100 (default)
        """
        self.config = lambda: 0

        self.config.max_degree = 1
        self.config.feature_dimension = 2
        self.config.num_of_class = 2
        
        self.config.params_init = 'normal'
        
        self.config.learning_rate = 1
        self.config.batch_size = 100
        
        """ override config with custom hyper-parameters if any. """
        for k, v in config.items():
            if getattr(self.config, k, None):
                setattr(self.config, k, v)
            else:
                raise AttributeError('Unknown hyper-parameter %s' % k)
        """ Attention This!! """
        self.config.space_mapping_process = (self.config.feature_dimension,
                                             self.config.num_of_class)
        
        """ `fp_trasformation`: feature_polynomial_transformation """
        self.fp_transformation = np.vstack(get_terms_collection(
            self.config.feature_dimension, self.config.max_degree))
        self.__init__params()
        
        
    # size & shape | init=`identity`; init=ENN.intializer.identity
    def __init__params(self):
        """
        `W`: coefficients_for_all_dimension_in_next_space
        len of `fp_transformation`: quantity_of_variable_term
        `b`: constant_coefficient_for_all_dimension_in_next_space
        """
        W_shape = len(self.fp_transformation), self.config.space_mapping_process[1]
        b_shape = self.config.space_mapping_process[1], 
        
        for case in switch(self.config.params_init):
            if case('identity'):
                self.W = np.zeros(W_shape)
                self.W[np.diag_indices(min(W_shape))] = 1
                self.b = np.zeros(b_shape)
                break
            if case('normal') \
            or case(np.random.normal):
                self.W = np.random.normal(size=W_shape)
                self.b = np.zeros(b_shape)
                break
            if case('random') \
            or case(np.random.random):
                self.W = np.random.random(W_shape)
                self.b = np.random.random(b_shape)
                break
            if case('ones') \
            or case(np.ones):
                self.W = np.ones(W_shape)
                self.b = 0.1 * np.ones(b_shape)
                break
            if case('defaults'):
                raise ValueError(
                    'illegal params initializer: %s!' % self.config.params_init)
            
    """ backpropagation """
    """ note that `x_s_train` is just an alias of `feature_input_s`. """
    def fit(self, x_s_train, y_s_train, steps=1000, \
            vars_trace_recording=False):
        x_s_train, y_s_train = np.array(x_s_train), np.array(y_s_train)
        trace_dict = {
            'W': [self.W.copy()], 
            'b': [self.b.copy()],
            'W_grad': [], 
            'b_grad': [],
            'loss': [self.loss(x_s_train, y_s_train, keepdims=True)]
        }
        
        sample_size = len(x_s_train)
        transformed_x_s_train = feature_polynomial_transform(
                    x_s_train, self.fp_transformation)
        print('transforming completed.')
        for step in range(steps):
            indices_batch = random.sample(
                range(sample_size), self.config.batch_size)

            """ i corresponds to row in `self.W` and j to col. """
            """ [IGNORE THIS] define them as function without actual object \
            reference can free memory at soon. and I think it's faster than \
            del statement. """
#             if 'transformed_x_s_train' not in locals():
                
#             else:
            x_s = x_s_train[indices_batch]
#             x_s = feature_polynomial_transform(x_s_train[indices_batch],
#                                                self.fp_transformation)
            y_minus_f_s = x_s.dot(self.W) + self.b - y_s_train[indices_batch]

            """ 
            1. `pd`: partial derivative 
            2. `l_on_f`: all the loss_j on the f_j
            3. `fj_on_w__j`: all the fj on the w_ij when j is fixed
            """
            pd_of_l_on_f_s = \
            2 / self.config.space_mapping_process[1] * y_minus_f_s
            pd_of_fj_on_w__j_s = x_s
            
            W_gradients = np.mean(
                pd_of_fj_on_w__j_s[:, :, np.newaxis] @ \
                pd_of_l_on_f_s[:, np.newaxis, :], axis=0
            )
            b_gradients = np.mean(pd_of_l_on_f_s, axis=0)
            
            """ this scaling has no physical meaning. """
#             W_gradients = W_gradients / (500*max(np.max(W_gradients), 
#                                             abs(np.min(W_gradients)))
#                                         )
#             b_gradients = b_gradients / (500*max(np.max(b_gradients), 
#                                             abs(np.min(b_gradients))) )

            scaling_factor_js = np.sum(W_gradients, axis=0) + b_gradients
            W_gradients = W_gradients / scaling_factor_js
            b_gradients = b_gradients / scaling_factor_js
            
#             import pdb
#             pdb.set_trace()
            self.W -= self.config.learning_rate * W_gradients
            self.b -= self.config.learning_rate * b_gradients
            
            if step % 5 == 0:
                print(self.loss(x_s_train, y_s_train), '|', 
                      self.accuracy(x_s_train, y_s_train))
            
            if vars_trace_recording:
                """ attention this! """
                trace_dict['W'].append(self.W.copy())
                trace_dict['b'].append(self.b.copy())
                trace_dict['W_grad'].append(W_gradients.copy())
                trace_dict['b_grad'].append(b_gradients.copy())
                trace_dict['loss'].append(
                    self.loss(x_s_train, y_s_train, keepdims=True))
                                
        print('training completed.') 
        for k in trace_dict:
            trace_dict[k] = np.array(trace_dict[k])
        return trace_dict

    
    # forward computing        
    def logits(self, x_s, W=None, b=None):
        assert (W is None) == (b is None)
        if W is not None and b is not None:
            return feature_polynomial_transform(x_s, self.fp_transformation).dot(W) +\
                   b
        else:
            return feature_polynomial_transform(x_s, self.fp_transformation).dot(self.W) +\
                   self.b

        
    def loss(self, x_s, y_s, W=None, b=None, keepdims=False):
        """
        return: 
            ndarray type with shape of (len(x_s), ).
        """
        x_s, y_s = np.array(x_s), np.array(y_s)
#         assert x_s.shape == y_s.shape
        assert (W is None) == (b is None)
#         import pdb
#         pdb.set_trace()
        if not keepdims:
            return np.mean(np.square(self.logits(x_s, W, b) - y_s))
        else:
            return np.square(self.logits(x_s, W, b) - y_s)

        
    def predict(self, x_s, W=None, b=None, one_hot=True):
        x_s = np.array(x_s)
        assert (W is None) == (b is None)
        logits = self.logits(x_s, W, b)
        max_indices = np.argmax(logits, 1)
        if not one_hot:
            predict = max_indices
        else:
            predict = np.zeros_like(logits).astype(np.int)
            predict[range(len(x_s)), max_indices] = 1
        return predict
    
    
    def accuracy(self, x_s, y_s, W=None, b=None):
        x_s, y_s = np.array(x_s), np.array(y_s)
#         assert x_s.shape == y_s.shape
        predict = self.predict(x_s, W, b, False)
        return np.mean(np.equal(predict, np.argmax(y_s, 1)))

    
if __name__ == '__main__':
    a = get_terms_collection(2, 2)
    """ 2000, 2 -> 2003000 """
    a = np.vstack(a)
    # print(a)
    print(len(a))