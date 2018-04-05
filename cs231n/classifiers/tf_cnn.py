import tensorflow as tf
import numpy as np

        
class Baseline(object):
    '''
    Baseline for experiments. 
    7 layer convolutional neural network
    (conv - relu)*6 - conv - softmax
    
    Inputs:
    - img: Input image of size (N, 32, 32, 3)
    - label: label associated with each input (N, C)
    N: dataset size
    C: number of classes
    '''
    def __init__(self, img, label):
        self.img = img
        self.label = label

        self.feature_maps = {}

        self.build_network()
        self.optimize()


    def block(self, input_data, scope, filters=16, conv_strides=1):
        with tf.variable_scope(scope):
            h = tf.layers.conv2d(input_data, filters=filters, kernel_size=3, strides=conv_strides, padding='valid', name='conv') # 30x30x64
            # nh = tf.layers.batch_normalization(h, training=is_training, name='bn')
            a = tf.nn.relu(h, name='relu')

        var_name = scope + '_h'
        self.feature_maps[var_name] = h
    
        return a

            
    def build_network(self):
        with tf.variable_scope('network'):
            a1 = self.block(tf.reshape(self.img, [-1,32,32,3]), 'layer1', filters=32)
            a2 = self.block(a1, 'layer2', filters=64)
            a3 = self.block(a2, 'layer3', filters=128, conv_strides=2)
            a4 = self.block(a3, 'layer4', filters=256)
            a5 = self.block(a4, 'layer5', filters=512)
            a6 = self.block(a5, 'layer6', filters=512, conv_strides=2)

            with tf.variable_scope('layer7'):
                h7 = tf.layers.conv2d(a6, filters=10, kernel_size=4, padding='valid', name='conv') # 1x1x10

            self.logits = tf.reshape(h7,[-1,10])

            
    def optimize(self, learning_rate=1e-3):
        with tf.name_scope('optimize'):
            with tf.name_scope('loss'):
                total_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.logits)
                self.mean_loss = tf.reduce_mean(total_loss)

            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean( tf.cast(tf.equal(tf.argmax(self.label, 1), tf.argmax(self.logits,1)), tf.float32) )

            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.mean_loss, global_step=self.global_step)


###################################### Aggregation ########################################################
            
class ConvModelMaxNormPool(Baseline):
    '''
    Max_norm layer is an aggregation layer which returns the 
    per-channel neighborhood value with the highest norm

    7 layer convolutional neural network
    (conv - max_norm)*6 - conv - softmax

    Inputs:
    - img: Input image of size (N, 32, 32, 3)
    - label: label associated with each input (N, C)
    N: dataset size
    C: number of classes
    '''
    def tf_max_norm(self, x, kernel_size=2):
        with tf.name_scope("max_norm"):
            N, _, _, C = x.get_shape().as_list() # (N, H, W, C)
            
            ksizes = [1, kernel_size, kernel_size, 1]
            strides = [1, 1, 1, 1]
            rates = [1, 1, 1, 1]
            
            patches = tf.extract_image_patches(x, ksizes, strides, rates, padding="SAME")
            _, nh, nw, D = patches.get_shape().as_list() # (N, num_patches_v, num_patches_h, D), D = C * kernel_size**2
            patches = tf.reshape(patches, (-1, nh, nw, kernel_size**2, C)) # (N, nh, nw, kernel_size**2, C)
            
            abs_patches = tf.abs(patches) # (N, nh, nw, kernel_size**2, C)
            
            abs_max = tf.cast(tf.argmax(abs_patches, dimension=3), tf.int32) # (N, nh, nw, C)
            abs_max = tf.reshape(abs_max, [-1, nh * nw * 1 * C])
            
            nh_ind, nw_ind, C_ind = tf.meshgrid(tf.range(nh), tf.range(nw), tf.range(C), indexing='ij')
            nh_ind, nw_ind, C_ind = tf.reshape(nh_ind, [-1]), tf.reshape(nw_ind, [-1]), tf.reshape(C_ind, [-1])
            
            out = tf.map_fn( lambda x: tf.gather_nd(x[0], tf.transpose(tf.stack([nh_ind, nw_ind, x[1], C_ind], 0), [1, 0])), (patches, abs_max), dtype=tf.float32 )
            out = tf.reshape(out, (-1, nh, nw, C))
        
        return out

    
    def block(self, input_data, scope, filters=16, conv_strides=1):
        with tf.variable_scope(scope):
            h = tf.layers.conv2d(input_data, filters=filters, kernel_size=3, strides=conv_strides, padding='valid', name='conv') # 30x30x64
            # nh = tf.layers.batch_normalization(h, training=is_training, name='bn')
            a = self.tf_max_norm(h)

        var_name = scope + '_h'
        self.feature_maps[var_name] = h
    
        return a

    
class ConvModelMaxPool(Baseline):
    '''
    Max_pool layer is an aggregation layer which returns the 
    per-channel neighborhood value with the highest value

    7 layer convolutional neural network
    (conv - max)*6 - conv - softmax

    Inputs:
    - img: Input image of size (N, 32, 32, 3)
    - label: label associated with each input (N, C)
    N: dataset size
    C: number of classes
    '''
    def block(self, input_data, scope, pool_size=2, pool_strides=1, filters=16, conv_strides=1):
        with tf.variable_scope(scope):
            h = tf.layers.conv2d(input_data, filters=filters, kernel_size=3, strides=conv_strides, padding='valid', name='conv') # 30x30x64
            # nh = tf.layers.batch_normalization(h, training=is_training, name='bn')
            a = tf.layers.max_pooling2d(h, pool_size=pool_size, strides=pool_strides, padding='same')

        var_name = scope + '_h'
        self.feature_maps[var_name] = h
    
        return a


class ConvModelAvgPool(Baseline):
    '''
    Sum_pool layer is an aggregation layer which returns the sum
    of per-channel neighborhood values

    7 layer convolutional neural network
    (conv - max)*6 - conv - softmax

    Inputs:
    - img: Input image of size (N, 32, 32, 3)
    - label: label associated with each input (N, C)
    N: dataset size
    C: number of classes
    '''
    def block(self, input_data, scope, pool_size=2, pool_strides=1, filters=16, conv_strides=1):
        with tf.variable_scope(scope):
            h = tf.layers.conv2d(input_data, filters=filters, kernel_size=3, strides=conv_strides, padding='valid', name='conv') # 30x30x64
            # nh = tf.layers.batch_normalization(h, training=is_training, name='bn')
            a = tf.nn.pool(h, window_shape=pool_size, pooling_type='AVG', padding='SAME', strides=pool_strides)

        var_name = scope + '_h'
        self.feature_maps[var_name] = h
    
        return a
    

#################################################### Nonlinearities ##########################################################
    
class ConvModelAbs(Baseline):
    '''
    Abs layer is a nonlinear activation unit that returns the absolute value (inspired by scattering networks)

    7 layer convolutional neural network
    (conv - abs)*6 - conv - softmax
    
    Inputs:
    - img: Input image of size (N, 32, 32, 3)
    - label: label associated with each input (N, C)
    N: dataset size
    C: number of classes
    '''
    def block(self, input_data, scope, filters=16, conv_strides=1):
        with tf.variable_scope(scope):
            h = tf.layers.conv2d(input_data, filters=filters, kernel_size=3, strides=conv_strides, padding='valid', name='conv') # 30x30x64
            # nh = tf.layers.batch_normalization(h, training=is_training, name='bn')
            a = tf.abs(h)

        var_name = scope + '_h'
        self.feature_maps[var_name] = h
    
        return a


class ConvModelSelection(Baseline):
    '''
    Selection layer is a nonlinear activation unit which returns 
    a vector of all zeros, except the channel with the highest 
    response in absolute value. This channel has the same (signed)
    value as the input

    7 layer convolutional neural network
    (conv - select)*6 - conv - softmax

    Inputs:
    - img: Input image of size (N, 32, 32, 3)
    - label: label associated with each input (N, C)
    N: dataset size
    C: number of classes
    '''
    def tf_selection(self, x):
        N, H, W, C = x.get_shape().as_list() # (N, H, W, C)
        
        abs_x = tf.abs(x) # (N, H, W, C)
        
        max_ind = tf.cast(tf.argmax(abs_x, dimension=3), tf.int32) # (N, H, W)
        # print(max_ind.get_shape().as_list())
        max_ind = tf.reshape(max_ind, [-1, H * W])
        # print(max_ind.get_shape().as_list())
        
        H_ind, W_ind = tf.meshgrid(tf.range(H), tf.range(W), indexing='ij')
        H_ind, W_ind = tf.reshape(H_ind, [-1]), tf.reshape(W_ind, [-1])
        
        out = tf.map_fn( lambda x: tf.gather_nd(x[0], tf.transpose(tf.stack([H_ind, W_ind, x[1]], 0), [1, 0])), (x, max_ind), dtype=tf.float32 )
        # print(out.get_shape().as_list())
        out = tf.reshape(out, (-1, H, W, 1))
        out = x * tf.cast(tf.equal(out, x), tf.float32)
        
        return out

    
    def block(self, input_data, scope, filters=16, conv_strides=1):
        with tf.variable_scope(scope):
            h = tf.layers.conv2d(input_data, filters=filters, kernel_size=3, strides=conv_strides, padding='valid', name='conv') # 30x30x64
            # nh = tf.layers.batch_normalization(h, training=is_training, name='bn')
            a = self.tf_selection(h)

        var_name = scope + '_h'
        self.feature_maps[var_name] = h
    
        return a


# TODO: ConvModelSelection version using max opposed to max_norm

# TODO: ConvModelSelection version with subsequent aggregation (sum perhaps)
# TODO: ConvModelAvg this aggregation approach is very close to linear, ConvModelSum would be
# TODO: ConvModelSum + Relu this approach does some aggregation but still uses the standard nonlinearity
# TODO: ConvModelSum + Abs

# Aggregation before and after the nonlinearity should be considered
# Batch normalization should also be considered

Networks = { 'avg':ConvModelAvgPool,
             'relu':Baseline,
             'max_norm':ConvModelMaxNormPool,
             'abs':ConvModelAbs,
             'max':ConvModelMaxPool,
             'select':ConvModelSelection }


if __name__ == '__main__':
    from cs231n.data_utils import Datasets

    datadir = 'cs231n/datasets/cifar-10-batches-py'
    dataset = Datasets['CIFAR10'](datadir)
    dataset.train_data = dataset.train_data.batch(64)

    tf.reset_default_graph()
    iterator = tf.data.Iterator.from_structure(dataset.train_data.output_types, dataset.train_data.output_shapes)
    img, label = iterator.get_next()
    relunet = Networks['relu'](img, label)

    tf.reset_default_graph()
    max_normiterator = tf.data.Iterator.from_structure(dataset.train_data.output_types, dataset.train_data.output_shapes)
    img, label = max_normiterator.get_next()
    max_normnet = Networks['max_norm'](img, label)

    tf.reset_default_graph()
    absiterator = tf.data.Iterator.from_structure(dataset.train_data.output_types, dataset.train_data.output_shapes)
    img, label = absiterator.get_next()
    absnet = Networks['abs'](img, label)

    tf.reset_default_graph()
    maxiterator = tf.data.Iterator.from_structure(dataset.train_data.output_types, dataset.train_data.output_shapes)
    img, label = maxiterator.get_next()
    maxnet = Networks['max'](img, label)

    tf.reset_default_graph()
    selectiterator = tf.data.Iterator.from_structure(dataset.train_data.output_types, dataset.train_data.output_shapes)
    img, label = selectiterator.get_next()
    selectnet = Networks['select'](img, label)
