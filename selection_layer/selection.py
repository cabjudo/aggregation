from builtins import range
import numpy as np
import tensorflow as tf


def tf_select_max(x):
    '''
    Selection layer (tensorflow implementation) is a nonlinear unit which takes a feature vector and returns the 
    a feature vector of all zeros except the index that had the maximum response
    '''
    # N, _, _, C = x.get_shape().as_list() # (N, C, H, W)
    N, C, H, W = x.get_shape().as_list() # (N, H, W, C)
    # print(x.get_shape().as_list())
    x = tf.transpose(x, perm=[0, 2, 3, 1]) # (N, H, W, C)
    
    max_ind = tf.cast(tf.argmax(x, dimension=3), tf.int32) # (N, H, W)
    # print(max_ind.get_shape().as_list())
    max_ind = tf.reshape(max_ind, [-1, H * W])
    # print(max_ind.get_shape().as_list())
    
    H_ind, W_ind = tf.meshgrid(tf.range(H), tf.range(W), indexing='ij')
    H_ind, W_ind = tf.reshape(H_ind, [-1]), tf.reshape(W_ind, [-1])

    out = tf.map_fn( lambda x: tf.gather_nd(x[0], tf.transpose(tf.stack([H_ind, W_ind, x[1]], 0), [1, 0])), (x, max_ind), dtype=tf.float32 )
    # print(out.get_shape().as_list())
    out = tf.reshape(out, (-1, H, W, 1))
    out = x * tf.cast(tf.equal(out, x), tf.float32)
    out = tf.transpose(out, [0, 3, 1, 2])
        
    return out


def tf_selection(x):
    '''
    Selection layer (tensorflow implementation) is a nonlinear unit which takes a feature vector and returns the 
    a feature vector of all zeros except the index that had the maximum (in absolute value) response
    '''
    # N, _, _, C = x.get_shape().as_list() # (N, C, H, W)
    N, C, H, W = x.get_shape().as_list() # (N, H, W, C)
    # print(x.get_shape().as_list())
    x = tf.transpose(x, perm=[0, 2, 3, 1]) # (N, H, W, C)
    
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
    out = tf.transpose(out, [0, 3, 1, 2])
        
    return out


def selection(x):
    '''
    Selection layer is a nonlinear unit which takes a feature vector and returns the 
    a feature vector of all zeros except the index that had the maximum (in absolute value) response
    '''
    N, C, H, W = x.shape # (N, C, H, W)
    x = np.transpose(x, (0, 2, 3, 1)) # (N, H, W, C)
    
    abs_x = np.abs(x) # (N, H, W, C)
    
    max_ind = np.argmax(abs_x, axis=-1)

    N_ind, H_ind, W_ind = np.meshgrid( np.arange(N), np.arange(H), np.arange(W))
    N_ind, H_ind, W_ind = N_ind.ravel(), H_ind.ravel(), W_ind.ravel()

    max_ind = max_ind[N_ind,H_ind,W_ind].ravel()
    
    out = np.zeros_like(x)
    out[N_ind, H_ind, W_ind, max_ind.ravel()] = x[N_ind, H_ind, W_ind, max_ind.ravel()]
    out = np.transpose(out, (0, 3, 1, 2))
    
    return out
