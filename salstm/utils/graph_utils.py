
import os.path
from itertools import compress

import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint


def print_tensors(checkpoint_file):
    # Code from https://stackoverflow.com/a/41917296
    # List ALL tensors example output: v0/Adam (DT_FLOAT) [3,3,1,80]
    print("\nCheckpoint:")
    inspect_checkpoint.print_tensors_in_checkpoint_file(
        file_name=checkpoint_file, tensor_name='', all_tensors=True)


def uninitialized_vars(session, variables, tag):
    is_not_initialized = session.run([~(tf.is_variable_initialized(var)) for var in variables])
    not_initialized_vars = list(compress(variables, is_not_initialized))

    print(f"Not initialized vars ({tag}) ({len(not_initialized_vars)})")
    if len(not_initialized_vars) > 0:
        print(*not_initialized_vars, sep='\n')

    return not_initialized_vars


def initialize_uninitialized_vars(session, default_scopes_list=None):
    """
    Reference:
        - https://stackoverflow.com/questions/44251666/how-to-initialize-tensorflow-variable-that-wasnt-saved-other-than-with-tf-globa
    Args:
        session: tf.Session
        default_scopes_list:

    Returns:
        - bool indicating if the model was restored or not

    """
    # Global variables
    not_initialized_vars = uninitialized_vars(session, tf.global_variables(), tag="global")

    if default_scopes_list is not None:
        # Builds list of variables that already must been initialized. Otherwise we may had a
        # problem in the initialization
        list_to_be_ignored = []
        for scope in default_scopes_list:
            current_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            list_to_be_ignored = list_to_be_ignored + current_vars

        default_vars_not_init = [var for var in not_initialized_vars if var in list_to_be_ignored]

        if not default_vars_not_init:
            print(f"The model was not properly restored. "
                  f"Variables {default_vars_not_init} were not initialized")
            return False

    # Local Variables
    local_not_initialized_vars = uninitialized_vars(session, tf.local_variables(), tag="local")

    if len(local_not_initialized_vars) > 0:
        print("Trying to initialize local variables...")
        session.run(tf.variables_initializer(local_not_initialized_vars))
        print("Trying to initialize local variables: Ok.\n")

    return True


def restore_model(saver, session, model_file=None, model_dir=None,
                  initialize_unvars=True, default_scopes_list=None,
                  latest_filename=None):
    """
    Restores the model from a directory containing the checkpoint..

    Args:
        saver: (tf.train.Saver)
        session: (tf.Session)
        model_file:
        model_dir:
        initialize_unvars:
        default_scopes_list:
        latest_filename: string - file with the list of most recent
        checkpoint filenames. Default value is usually "checkpoint"

    Returns:

    """
    print("Model_file={}, Model_dir={}".format(model_file, model_dir))

    if model_file is not None and not os.path.isfile(model_file):
        model_file = None

    save_path = "<NONE>"
    if model_file is not None:
        saver.restore(session, model_file)
    else:
        save_path = tf.train.latest_checkpoint(model_dir, latest_filename=latest_filename)
        saver.restore(session, save_path)
    print("Model restored from: %s" % (model_file if model_file is not None else save_path))

    # Initialize the uninitialized variables
    vars_ok = True
    if initialize_unvars:
        vars_ok = initialize_uninitialized_vars(session, default_scopes_list)

    return vars_ok
