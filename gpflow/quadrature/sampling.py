


def ndiag_mc(funcs, S: int, Fmu, Fvar, logspace: bool = False, epsilon=None, **Ys):
    """
    Computes N Gaussian expectation integrals of one or more functions
    using Monte Carlo samples. The Gaussians must be independent.

    :param funcs: the integrand(s):
        Callable or Iterable of Callables that operates elementwise
    :param S: number of Monte Carlo sampling points
    :param Fmu: array/tensor
    :param Fvar: array/tensor
    :param logspace: if True, funcs are the log-integrands and this calculates
        the log-expectation of exp(funcs)
    :param **Ys: arrays/tensors; deterministic arguments to be passed by name

    Fmu, Fvar, Ys should all have same shape, with overall size `N`
    :return: shape is the same as that of the first Fmu
    """
    N, D = Fmu.shape[0], Fvar.shape[1]

    if epsilon is None:
        epsilon = tf.random.normal((S, N, D), dtype=default_float())

    mc_x = Fmu[None, :, :] + tf.sqrt(Fvar[None, :, :]) * epsilon
    mc_Xr = tf.reshape(mc_x, (S * N, D))

    for name, Y in Ys.items():
        D_out = Y.shape[1]
        # we can't rely on broadcasting and need tiling
        mc_Yr = tf.tile(Y[None, ...], [S, 1, 1])  # [S, N, D]_out
        Ys[name] = tf.reshape(mc_Yr, (S * N, D_out))  # S * [N, _]out

    def eval_func(func):
        feval = func(mc_Xr, **Ys)
        feval = tf.reshape(feval, (S, N, -1))
        if logspace:
            log_S = tf.math.log(to_default_float(S))
            return tf.reduce_logsumexp(feval, axis=0) - log_S  # [N, D]
        else:
            return tf.reduce_mean(feval, axis=0)

    if isinstance(funcs, Iterable):
        return [eval_func(f) for f in funcs]
    else:
        return eval_func(funcs)
