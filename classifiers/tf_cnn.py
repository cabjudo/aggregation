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
    def __init__(self, img, label, arch):
        self.img = img
        self.label = label
        self.arch = arch

        self.feature_maps = {}

        self.build_network()


    def block(self, input_data, scope, filters=16, conv_strides=1, kernel_size=3, padding='valid'):
        with tf.variable_scope(scope):
            h = tf.layers.conv2d(input_data, filters=filters, kernel_size=kernel_size, strides=conv_strides, padding=padding, name='conv')
            # nh = tf.layers.batch_normalization(h, training=is_training, name='bn')
            a = tf.nn.relu(h, name='relu')

        var_name = scope + '_h'
        self.feature_maps[var_name] = h
    
        return a

            
    def build_network(self):
        num_layers = len(self.arch['filters'])
        
        with tf.variable_scope('network'):
            input_data = tf.reshape(self.img, [-1] + self.arch['indim'] )
            for l in range(num_layers - 1):
                layer_name = 'layer' + str(l + 1)
                filters = self.arch['filters'][l]
                kernel_size = self.arch['kernel_size'][l]
                conv_strides= self.arch['strides'][l]
                padding = self.arch['padding'][l]

                input_data = self.block(input_data, layer_name, filters=filters, conv_strides=conv_strides, kernel_size=kernel_size, padding=padding)

            # classification layer
            l = num_layers - 1
            layer_name = 'layer' + str(l + 1)
            filters = self.arch['filters'][l]
            kernel_size = self.arch['kernel_size'][l]
            conv_strides= self.arch['strides'][l]
            padding = self.arch['padding'][l]
                
            with tf.variable_scope(layer_name):
                out = tf.layers.conv2d(input_data, filters=filters, kernel_size=kernel_size, padding=padding, name='conv')

            self.logits = tf.reshape(out,[-1,10])


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

    
    def block(self, input_data, scope, filters=16, conv_strides=1, kernel_size=3, padding='valid'):
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
    def block(self, input_data, scope, pool_size=2, pool_strides=1, filters=16, conv_strides=1, kernel_size=3, padding='valid'):
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
    def block(self, input_data, scope, pool_size=2, pool_strides=1, filters=16, conv_strides=1, kernel_size=3, padding='valid'):
        with tf.variable_scope(scope):
            h = tf.layers.conv2d(input_data, filters=filters, kernel_size=3, strides=conv_strides, padding='valid', name='conv') # 30x30x64
            # nh = tf.layers.batch_normalization(h, training=is_training, name='bn')
            a = tf.layers.average_pooling2d(h, pool_size=pool_size, strides=pool_strides, padding='same')

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
    def block(self, input_data, scope, filters=16, conv_strides=1, kernel_size=3, padding='valid'):
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

    
    def block(self, input_data, scope, filters=16, conv_strides=1, kernel_size=3, padding='valid'):
        with tf.variable_scope(scope):
            h = tf.layers.conv2d(input_data, filters=filters, kernel_size=3, strides=conv_strides, padding='valid', name='conv') # 30x30x64
            # nh = tf.layers.batch_normalization(h, training=is_training, name='bn')
            a = self.tf_selection(h)

        var_name = scope + '_h'
        self.feature_maps[var_name] = h
    
        return a

    
class ConvModelSelectMax(Baseline):
    '''
    Select_max layer is a nonlinear activation unit which returns 
    a vector of all zeros, except the channel with the highest 
    response. This channel has the same (signed)
    value as the input

    7 layer convolutional neural network
    (conv - select)*6 - conv - softmax

    Inputs:
    - img: Input image of size (N, 32, 32, 3)
    - label: label associated with each input (N, C)
    N: dataset size
    C: number of classes
    '''
    def tf_select_max(self, x):
        '''
        Selection layer (tensorflow implementation) is a nonlinear unit which takes a feature vector and returns the 
        a feature vector of all zeros except the index that had the maximum response
        '''
        N, H, W, C = x.get_shape().as_list() # (N, H, W, C)
        
        max_ind = tf.cast(tf.argmax(x, dimension=3), tf.int32) # (N, H, W)
        # print(max_ind.get_shape().as_list())
        max_ind = tf.reshape(max_ind, [-1, H * W])
        # print(max_ind.get_shape().as_list())
        
        H_ind, W_ind = tf.meshgrid(tf.range(H), tf.range(W), indexing='ij')
        H_ind, W_ind = tf.reshape(H_ind, [-1]), tf.reshape(W_ind, [-1])
        
        out = tf.map_fn( lambda x: tf.gather_nd(x[0], tf.transpose(tf.stack([H_ind, W_ind, x[1]], 0), [1, 0])), (x, max_ind), dtype=tf.float32 )
        out = tf.reshape(out, (-1, H, W, 1))
        out = x * tf.cast(tf.equal(out, x), tf.float32)
        
        return out

    
    def block(self, input_data, scope, filters=16, conv_strides=1, kernel_size=3, padding='valid'):
        with tf.variable_scope(scope):
            h = tf.layers.conv2d(input_data, filters=filters, kernel_size=3, strides=conv_strides, padding='valid', name='conv')
            # nh = tf.layers.batch_normalization(h, training=is_training, name='bn')
            a = self.tf_select_max(h)

        var_name = scope + '_h'
        self.feature_maps[var_name] = h
    
        return a

    
# TODO: ConvModelSelection version with subsequent aggregation (sum perhaps)
# TODO: ConvModelSum + Relu this approach does some aggregation but still uses the standard nonlinearity
# TODO: ConvModelSum + Abs

# Aggregation before and after the nonlinearity should be considered
# Batch normalization should also be considered

Networks = { 'avg':ConvModelAvgPool,
             'relu':Baseline,
             'max_norm':ConvModelMaxNormPool,
             'abs':ConvModelAbs,
             'max':ConvModelMaxPool,
             'select':ConvModelSelection,
             'select_max':ConvModelSelectMax }


if __name__ == '__main__':
    from convnet.data_utils import Datasets

    from convnet.util.config_reader import get_model
    from convnet.util.config_reader import get_dataset
    from convnet.util.config_reader import get_training

    configfile = '/home/christine/projects/convnet/config/default.ini'
    model_info = get_model(configfile)
    dataset_info = get_dataset(configfile)
    training_info = get_training(configfile)

    dataset = Datasets['CIFAR10'](dataset_info)
    dataset.train_data = dataset.train_data.batch(training_info['options']['batch_size'])

    for key,value in Networks.items():
        tf.reset_default_graph()
        iterator = tf.data.Iterator.from_structure(dataset.train_data.output_types, dataset.train_data.output_shapes)
        img, label = iterator.get_next()
        relunet = Networks[key](img, label, arch=model_info['params'])
        print(key + ' completed')
