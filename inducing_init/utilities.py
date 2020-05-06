import tensorflow as tf


def set_trainable(model: tf.Module, flag: bool):
    """
    Set trainable flag for all `tf.Variable`s and `gpflow.Parameter`s in a module.
    """
    for variable in model.variables:
        if "jitter" not in variable.name:
            variable._trainable = flag
