import logging

import tensorflow as tf

def nested_tuple_apply(t, fn, *a, **k):
    if isinstance(t, tuple):
        return tuple(nested_tuple_apply(e, fn, *a, **k) for e in t)
    else:
        return fn(t, *a, **k)

def parallel_nested_tuples_apply(ts, fn, *a, **k):
    """ Like `nested_tuple_apply` but assumes that `fn` takes multiple arguments.
    If `ts` is a list of tensors then the function returns `fn(*ts, *a, **k)`.
    If `ts` is a list of nested tuples then the function returns nested tuple of tensors `fn(*leaves, *a, **k)` where leaves is a list of tensors sharing the same position in nested tuples form `ts` (this will be also the position of result in returned nested tuple).
    Args: 
        ts: list of tensors or iterable of (possibly nested) tuples; if elements are nested tuples then they have to have the same structure.
        fn: a function taking at least as much arguments as there are elements of `ts`.
        a: iterable of additional positional arguments passed to fn (after arguments from elements of `ts`).
        kw: mapping of additional keyword arguments passed to fn.
    Returns:
        Tensor or nested tuple. If nested tuple then exact types (sublcasses) of returned tuple are the same as in first element of `ts`
    """
    ts = [*ts]
    is_tuple = [isinstance(e, tuple) for e in ts]
    if all(is_tuple):
        if not len(ts):
            return fn(*a, **k)
        tuple_type = type(next(iter(ts)))
        lengths = [len(es) for es in ts]
        if not all(l != 0 for l in lengths):
            raise ValueError("Empty tuples are not allowed in parallel_nested_tuples_apply. Invalid structure: {}".format(ts))
        if not all(a == b for a, b in zip(lengths[:-1], lengths[1:])):
            raise ValueError("Structures of nested tuples are not identical. Invalid structure: {}".format(ts))
        print("====")
        print(ts)
        print([*zip(*ts)])
        return tuple_type(parallel_nested_tuples_apply(es, fn, *a, **k) for es in zip(*ts))
    elif any(is_tuple):
        raise ValueError("Structures of nested tuples form ts don't match.")
    else:
        return fn(*ts, *a, **k)

def top_k_from_2d_tensor(tensor2d, k):
        """Find top k values of 2D tensor. Return values themselves and vectors their first and second indices.
        If two elements are equal, the lower-row has priority, if they are in the same row, lower index has priority. """
        flat_tensor = tf.reshape(tensor2d, (-1,))
        top_values, top_indices = tf.nn.top_k(flat_tensor, k)
        top_index1 = top_indices // tf.shape(tensor2d)[1]
        top_index2 = top_indices % tf.shape(tensor2d)[1]
        return top_values, (top_index1, top_index2)

def batched_top_k_from_2d_tensor(tensor2d, k):
        """Find top k values of 2D tensor. Return values themselves and vectors their first and second indices.
        If two elements are equal, the lower-row has priority, if they are in the same row, lower index has priority. """
        batch_size = tf.shape(tensor2d)[0]
        flat_tensor = tf.reshape(tensor2d, (batch_size, -1,))
        top_values, top_indices = tf.nn.top_k(flat_tensor, k)
        top_index1 = top_indices // tf.shape(tensor2d)[-1]
        top_index2 = top_indices % tf.shape(tensor2d)[-1]
        return top_values, (top_index1, top_index2)


def repeat_in_ith_dimension(tensor, i, k):
    """Adds new dimension (namely i-th dimension) and repeats original tensor k times along this dimesion.
    
    output[a_0, ..., a_(i-1), a_i, a_(i+1), ..., a_(n-i)] = input[a_0, ..., a_(i-1), a_(i+1), ..., a_(n-1)] 
    
    Args:
        tensor (tf.Tensor): input tensor
        i: index of new dimension
        k: size in i-th dimension (i.e. how many times the original vector will be repeated)
    
    Returns:
        tf.Tensor of the same dtype and size [s_0, ..., s_(i-1), k, s_(i+1), ..., s_(n-1)]
    """
    expanded = tf.expand_dims(tensor, i)
    multiplicity = [1 for _ in expanded.shape]
    multiplicity[i] = k
    return tf.tile(expanded, multiplicity)


def inject_hparams(target_object, hparams, properties):
    for param_name in properties:
        param_value = hparams[param_name]
        setattr(target_object, param_name, param_value)

def maybe_inject_hparams(target_object, hparams, properties):
    for param_name in properties:
        current_value = getattr(target_object, param_name, default=None)
        if current_value is not None:
            param_value = hparams[param_name]
            logging.debug("Setting param {} = {} in {}".format(param_name, param_value, target_object))
            setattr(target_object, param_name, param_value)
        else:
            logging.debug("Param already set, leaving as is: {} = {} in {}".format(param_name, param_value, target_object))

class without_context_manager:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass
without = without_context_manager()