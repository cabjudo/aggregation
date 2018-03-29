from __future__ import print_function
from builtins import range

import numpy as np
import tensorflow as tf

from max_norm_layer.max_norm import max_norm_forward
from max_norm_layer.max_norm import max_norm_backward
from max_norm_layer.max_norm import tf_max_norm
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient


def test_tf_max_norm():
    x_shape = (2, 3, 4, 4)
    x_size = np.prod(x_shape)
    x = tf.reshape( tf.linspace(-0.3, 0.4, x_size), x_shape )
    out = tf_max_norm(x)
    
    correct_out = tf.constant([[[[-0.3,        -0.29263158, -0.28526316 ],
                                 [-0.27052632, -0.26315789, -0.25578947 ],
                                 [-0.24105263, -0.23368421, -0.22631579 ]],
                                
                                [[-0.18210526, -0.17473684, -0.16736842 ],
                                 [-0.15263158, -0.14526316, -0.13789474 ],
                                 [-0.12315789, -0.11578947, -0.10842105 ]],
                                
                                [[-0.06421053, -0.05684211, -0.04947368 ],
                                 [-0.03473684, -0.02736842, -0.02       ],
                                 [ 0.03157895,  0.03894737,  0.04631579 ]]],
                               
                               
                               [[[ 0.09052632,  0.09789474,  0.10526316],
                                 [ 0.12,        0.12736842,  0.13473684],
                                 [ 0.14947368,  0.15684211,  0.16421053]],
                                
                                [[ 0.20842105,  0.21578947,  0.22315789],
                                 [ 0.23789474,  0.24526316,  0.25263158],
                                 [ 0.26736842,  0.27473684,  0.28210526]],
                                
                                [[ 0.32631579,  0.33368421,  0.34105263],
                                 [ 0.35578947,  0.36315789,  0.37052632],
                                 [ 0.38526316,  0.39263158,  0.4       ]]]])


    with tf.Session() as sess:
        c, max_norm_out = sess.run([correct_out, out])

    assert np.allclose(c, max_norm_out)
        

def test_max_norm_forward():
    x_shape = (2, 3, 4, 4)
    x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
    pool_param = {'pool_width': 2, 'pool_height': 2}
    out, _ = max_norm_forward(x, pool_param)
    
    correct_out = np.array([[[[-0.3,        -0.29263158, -0.28526316 ],
                              [-0.27052632, -0.26315789, -0.25578947 ],
                              [-0.24105263, -0.23368421, -0.22631579 ]],
                              
                              [[-0.18210526, -0.17473684, -0.16736842 ],
                               [-0.15263158, -0.14526316, -0.13789474 ],
                               [-0.12315789, -0.11578947, -0.10842105 ]],
                              
                              [[-0.06421053, -0.05684211, -0.04947368 ],
                               [-0.03473684, -0.02736842, -0.02       ],
                               [ 0.03157895,  0.03894737,  0.04631579 ]]],
                             
                             
                             [[[ 0.09052632,  0.09789474,  0.10526316],
                               [ 0.12,        0.12736842,  0.13473684],
                               [ 0.14947368,  0.15684211,  0.16421053]],
                              
                              [[ 0.20842105,  0.21578947,  0.22315789],
                               [ 0.23789474,  0.24526316,  0.25263158],
                               [ 0.26736842,  0.27473684,  0.28210526]],
                              
                              [[ 0.32631579,  0.33368421,  0.34105263],
                               [ 0.35578947,  0.36315789,  0.37052632],
                               [ 0.38526316,  0.39263158,  0.4       ]]]])


    assert np.allclose(out, correct_out)


def test_max_norm_backward():
    np.random.seed(231)
    x = np.random.randn(3, 2, 8, 8)
    dout = np.random.randn(3, 2, 7, 7)
    pool_param = {'pool_height': 2, 'pool_width': 2}
    
    dx_num = eval_numerical_gradient_array(lambda x: max_norm_forward(x, pool_param)[0], x, dout)
    
    out, cache = max_norm_forward(x, pool_param)
    dx = max_norm_backward(dout, cache)

    assert np.allclose(dx, dx_num)
