from keras import backend as K
from keras import initializers
from keras.engine.topology import Layer
from keras.layers.core import *

import tensorflow as tf
#from keras.backend.cntk_backend import dtype

class GraphConv(Layer):
    '''Convolution operator for graphs.

    REQUIRES THEANO BACKEND (line 130).
	
    Implementation reduce the convolution to tensor product, 
    as described in "A generalization of Convolutional Neural 
    Networks to Graph-Structured Data".  

    When using this layer as the first layer in a model,
    provide an `input_shape` argument
    (tuple of integers e.g. `(1000, 1)` for a graph 
    with 1000 features (or nodes) and a single filter.

    # Arguments
        filters: Number of convolution kernels to use
            (dimensionality of the output).
	   num_neighbors: the number of neighbors the convolution
            would be applied on (analogue to filter length)
        neighbors_ix_mat: A matrix with dimensions
            (variables, num_neighbors) where the entry [Q]_ij
            denotes for the i's variable the j's closest neighbor.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        use_bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: Number of filters/dimensions in the input.
            Either this argument or the keyword argument `input_shape`must be
            provided when using this layer as the first layer in a model.
        input_length: Length of input sequences, when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
            
    # Input shape    
        3D tensor with shape:
        `(batch_size, features, input_dim)`.
    # Output shape
        3D tensor with shape:
        `(batch_size, features, filters)`.
    '''
        
    def __init__(self, 
                 filters, 
                 num_neighbors,
                 neighbors_ix_mat, 
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None, 
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,                 
                 **kwargs):

        if K.backend() != 'tensorflow':
            raise Exception("GraphConv with Tensorflow Backend.")
            
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
           kwargs['input_shape'] = (kwargs.pop('input_dim'),)
           
        super(GraphConv, self).__init__(**kwargs)        
      
        self.filters = filters     
        self.num_neighbors = num_neighbors
        self.neighbors_ix_mat = neighbors_ix_mat
        
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=3)      

    def build(self, input_shape):
        input_dim = input_shape[2]
        kernel_shape = (self.num_neighbors, input_dim, self.filters)
 
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True
                       
    def call(self, x):
        '''The idea: 
           1. associate all the features to each feature position of a training instance
           2. Use the top-k feature indices provided by self.neighbors_ix_mat to shrink the third dimension from #features to #Top_K features
           
           Input x: <batch, #features, 1> - It is converted to <batch, #features, #features> - and then flattened to 'tiled' variable
           self.neighbor_ix_mat provides top-k feats for each feature position, we want to do this for the full batch - so we tile it for batch size first
           Once tiled, we update integer index ids for each batch size and each feature position such that - when it is flattened
               It can be used to gather the top-k feat values from 'tiled'
           We then gather and reshape x with the neighbor info: This is stored in x_expanded
               x_expanded = <batch, #features, #top-k_features>
        '''
#        x_shape=K.shape(x)
#        x_s=tf.transpose(x, perm=[0,2,1])
#        x_s=tf.expand_dims(x_s, 2)
#        tiled=tf.reshape(tf.tile(x_s, [1,1,x_shape[1],1]), (-1,1))
#        
#        nei_expanded=tf.tile(tf.expand_dims(tf.constant(self.neighbors_ix_mat), 0), (tf.shape(x)[0],1,1))
#        nei_expanded=tf.reshape(tf.tile(tf.expand_dims(nei_expanded, 1), [1,1,x_shape[2],1]), (-1,self.num_neighbors))
#        
#        r=tf.tile(tf.expand_dims(tf.range(0, x_shape[0]*x_shape[1]*x_shape[2]), 1), [1, self.num_neighbors])
#        idx=tf.reshape(tf.to_int64(r)*tf.to_int64(x_shape[0]) + tf.to_int64(nei_expanded), (-1,1))
#
#        x_expanded=tf.gather(tiled, idx)
#        x_expanded=tf.reshape(x_expanded, (x_shape[0], x_shape[2], x_shape[1], -1))
#        x_expanded=tf.transpose(x_expanded, perm=[0,2,3,1])

        #Tensor dot implementation with tensorflow
        #print(self.neighbors_ix_mat)
        x_expanded = tf.gather(x, self.neighbors_ix_mat, axis=1)
        #x_expanded = x[:,self.neighbors_ix_mat,:]
        output = tf.tensordot(x_expanded, self.kernel, [[2,3],[0,1]])   
        if self.use_bias:
            output += tf.reshape(self.bias, (1, 1, self.filters))
        
        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.filters)

