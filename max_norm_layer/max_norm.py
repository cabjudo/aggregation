from builtins import range
import numpy as np
import tensorflow as tf


def tf_max_norm(x, kernel_size=2):
    # N, _, _, C = x.get_shape().as_list() # (N, C, H, W)
    N, C, _, _ = x.get_shape().as_list() # (N, H, W, C)
    # print(x.get_shape().as_list())
    x = tf.transpose(x, perm=[0, 2, 3, 1]) # (N, H, W, C)
    
    ksizes = [1, kernel_size, kernel_size, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    
    patches = tf.extract_image_patches(x, ksizes, strides, rates, padding="VALID")
    _, nh, nw, D = patches.get_shape().as_list() # (N, num_patches_v, num_patches_h, D), D = C * kernel_size**2
    patches = tf.reshape(patches, (-1, nh, nw, kernel_size**2, C)) # (N, nh, nw, kernel_size**2, C)
    # print(patches.get_shape().as_list())
    abs_patches = tf.abs(patches) # (N, nh, nw, kernel_size**2, C)
    
    abs_max = tf.cast(tf.argmax(abs_patches, dimension=3), tf.int32) # (N, nh, nw, C)
    # abs_max = tf.reshape( abs_max, (-1, nh, nw, 1, C) )
    abs_max = tf.reshape(abs_max, [-1, nh * nw * 1 * C])
    # print(abs_max.get_shape().as_list())
    
    nh_ind, nw_ind, C_ind = tf.meshgrid(tf.range(nh), tf.range(nw), tf.range(C), indexing='ij')
    nh_ind, nw_ind, C_ind = tf.reshape(nh_ind, [-1]), tf.reshape(nw_ind, [-1]), tf.reshape(C_ind, [-1])

    out = tf.map_fn( lambda (x,y): tf.gather_nd(x, tf.transpose(tf.stack([nh_ind, nw_ind, y, C_ind], 0), [1, 0])), (patches, abs_max), dtype=tf.float32 )
    # print(out.get_shape().as_list())
    out = tf.reshape(out, (-1, nh, nw, C))
    out = tf.transpose(out, [0, 3, 1, 2])
        
    return out

def tf_max_norm_old(x, kernel_size=2):
    N, C, _, _ = x.get_shape().as_list() # (N, C, H, W)
    x = tf.transpose(x, perm=[0, 2, 3, 1]) # (N, H, W, C)

    ksizes = [1, kernel_size, kernel_size, 1]
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    
    patches = tf.extract_image_patches(x, ksizes, strides, rates, padding="VALID")
    _, nh, nw, D = patches.get_shape().as_list() # (N, num_patches_v, num_patches_h, D), D = C * kernel_size**2
    patches = tf.reshape(patches, (N, nh, nw, kernel_size**2, C)) # (N, nh, nw, kernel_size**2, C)
    abs_patches = tf.abs(patches) # (N, nh, nw, kernel_size**2, C)

    abs_max = tf.cast(tf.argmax(abs_patches, dimension=3), tf.int32) # (N, nh, nw, C)
    abs_max = tf.reshape( abs_max, (N, nh, nw, 1, C) )
    abs_max = tf.reshape(abs_max, [-1])
    
    N_ind, nh_ind, nw_ind, C_ind = tf.meshgrid(tf.range(N), tf.range(nh), tf.range(nw), tf.range(C), indexing='ij')
    N_ind, nh_ind, nw_ind, C_ind = tf.reshape(N_ind, [-1]), tf.reshape(nh_ind, [-1]), tf.reshape(nw_ind, [-1]), tf.reshape(C_ind, [-1])
    ind = tf.transpose(tf.pack([N_ind, nh_ind, nw_ind, abs_max, C_ind], 0), [1, 0])

    out = tf.gather_nd(patches, ind)
    out = tf.reshape(out, (N, nh, nw, C))
    out = tf.transpose(out, [0, 3, 1, 2])

    return out


def max_mag_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max norm layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    
    N, C, H, W = x.shape
    WW, HH = pool_param['pool_height'], pool_param['pool_width']

    W_out = np.int( W - WW + 1 )
    H_out = np.int( H - HH + 1 )

    out = np.zeros((N, C, H_out, W_out))
    
    for r in range(H_out):
        for col in range(W_out):
            r_end, c_end = r + HH, col + WW

            # x with spatial flattened
            x_ = x[:, :, r:r_end, col:c_end ].reshape(N, C, -1)
            abs_x = np.absolute(x_)

            # get indices of max norm elements
            ind = np.argmax(abs_x, axis=-1)

            C_ind, N_ind = np.meshgrid(range(C), range(N))
            N_ind, C_ind = N_ind.ravel(), C_ind.ravel()
            out[N_ind, C_ind, r, col] = x_[N_ind, C_ind, ind.ravel()]
            
    cache = (x, pool_param)
    return out, cache


def max_mag_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None

    x, pool_param = cache
    N, C, H, W = x.shape
    N, C, H_out, W_out = dout.shape
    WW, HH = pool_param['pool_height'], pool_param['pool_width']
    
    dx = np.zeros(x.shape)

    for r in np.arange(H_out):
        for col in np.arange(W_out):
            r_end, c_end = r + HH, col + WW

            # x with spatial flattened
            x_ = x[:, :, r:r_end, col:c_end ].reshape(N, C, -1)
            abs_x = np.absolute(x_)

            # get indices of max norm elements
            ind = np.argmax(abs_x, axis=2)

            C_ind, N_ind = np.meshgrid(range(C), range(N))
            N_ind, C_ind = N_ind.ravel(), C_ind.ravel()

            mask = np.equal(x_, np.tile(x_[N_ind, C_ind, ind.ravel()].reshape(N,C,1),(1,1,1,HH*WW))).reshape(N,C,HH,WW)

            dx[:, :, r:r_end, col:c_end][mask] += dout[:, :, r, col].ravel()

    return dx


def max_norm_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max norm layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    
    N, C, H, W = x.shape
    WW, HH = pool_param['pool_height'], pool_param['pool_width']

    W_out = np.int( W - WW + 1 )
    H_out = np.int( H - HH + 1 )

    out = np.zeros((N, C, H_out, W_out))
    
    for r in range(H_out):
        for col in range(W_out):
            r_end, c_end = r + HH, col + WW

            # x with spatial flattened
            x_ = x[:, :, r:r_end, col:c_end ].reshape(N, C, -1)
            abs_x = np.absolute(x_)

            # get indices of max norm elements
            ind = np.argmax(abs_x, axis=-1)

            C_ind, N_ind = np.meshgrid(range(C), range(N))
            N_ind, C_ind = N_ind.ravel(), C_ind.ravel()
            out[N_ind, C_ind, r, col] = x_[N_ind, C_ind, ind.ravel()]
            
    cache = (x, pool_param)
    return out, cache


def max_norm_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None

    x, pool_param = cache
    N, C, H, W = x.shape
    N, C, H_out, W_out = dout.shape
    WW, HH = pool_param['pool_height'], pool_param['pool_width']
    
    dx = np.zeros(x.shape)

    for r in np.arange(H_out):
        for col in np.arange(W_out):
            r_end, c_end = r + HH, col + WW

            # x with spatial flattened
            x_ = x[:, :, r:r_end, col:c_end ].reshape(N, C, -1)
            abs_x = np.absolute(x_)

            # get indices of max norm elements
            ind = np.argmax(abs_x, axis=2)

            C_ind, N_ind = np.meshgrid(range(C), range(N))
            N_ind, C_ind = N_ind.ravel(), C_ind.ravel()

            mask = np.equal(x_, np.tile(x_[N_ind, C_ind, ind.ravel()].reshape(N,C,1),(1,1,1,HH*WW))).reshape(N,C,HH,WW)

            dx[:, :, r:r_end, col:c_end][mask] += dout[:, :, r, col].ravel()

    return dx

