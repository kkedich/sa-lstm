import tensorflow as tf

def transpose_batch_time(x):
    """
    Code from
    https://github.com/tensorflow/tensorflow/blob/a6d8ffae097d0132989ae4688d224121ec6d8f35/tensorflow/python/ops/rnn.py#L44

    Transposes the batch and time dimensions of a Tensor.
    If the input tensor has rank < 2 it returns the original tensor. Retains as
    much of the static shape information as possible.
    Args:
      x: A Tensor.
    Returns:
      x transposed along the first two dimensions.
    """
    x_static_shape = x.get_shape()
    if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
        return x

    x_rank = tf.rank(x)
    x_t = tf.transpose(x, tf.concat(([1, 0], tf.range(2, x_rank)), axis=0))
    x_t.set_shape(tf.TensorShape(
        [x_static_shape[1].value, x_static_shape[0].value]).concatenate(x_static_shape[2:]))

    return x_t