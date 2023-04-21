import torch
import functools
import numpy as np


def grad_of_gaussiankernel(x, sigma):
    grad_of_gauss = -(x / sigma ** 2) * calc_gauss(x, mu=0.0, sigma=sigma)
    return grad_of_gauss


def calc_gauss(x, mu=0.0, sigma=1.0):
    return 1.0 / (sigma * (2.0 * np.pi)**0.5) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)


def mc_estimate(f_xi, p_xi):
    N = f_xi.shape[0]
    estimate = 1. / N * (f_xi / p_xi).sum(dim=0)  # average along batch axis, leave dimension axis unchanged
    return estimate


def convolve(kernel_fn, render_fn, importance_fn, theta, nsamples, context_args, *args):
    # sample, get kernel(samples), get render(samples), return mc estimate of output
    # expect theta to be of shape [1, n], where n is dimensionality

    dim = theta.shape[-1]
    sigma = context_args['sigma']
    update_fn = context_args['update_fn']  # fn pointer to e.g. apply_rotation

    if context_args['sampler'] == 'uniform':
        raise NotImplementedError("for now only IS sampler supported")

    # get importance-sampled taus
    tau, pdf = importance_fn(nsamples, sigma, context_args['antithetic'], dim, context_args['device'])

    # get kernel weight at taus
    weights = kernel_fn(tau, sigma)

    # twice as many samples when antithetic
    if context_args['antithetic']:
        nsamples *= 2

    # shift samples around current parameter
    theta_p = torch.cat([theta] * nsamples, dim=0) - tau

    renderings, avg_img = render_fn(theta_p, update_fn, context_args)    # output shape [N]

    # weight output by kernel, mc-estimate gradient
    output = renderings.unsqueeze(-1) * weights
    forward_output = mc_estimate(output, pdf)

    return forward_output, avg_img


def importance_gradgauss(n_samples, sigma, is_antithetic, dim, device):
    eps = 0.00001
    randoms = torch.rand(n_samples, dim).to(device)

    def icdf(x, sigma):
        res = torch.zeros_like(x).to(device)
        res[mask == 1] = torch.sqrt(-2.0 * sigma ** 2 * torch.log(2.0 * (1.0 - x[mask == 1])))
        res[mask == -1] = torch.sqrt(-2.0 * sigma ** 2 * torch.log(2.0 * x[mask == -1]))
        return res

    # samples and AT samples
    if is_antithetic:
        randoms = torch.cat([randoms, 1.0 - randoms])

    # avoid NaNs bc of numerical instabilities in log
    randoms[torch.isclose(randoms, torch.ones_like(randoms))] -= eps
    randoms[torch.isclose(randoms, torch.zeros_like(randoms))] += eps
    randoms[torch.isclose(randoms, torch.full_like(randoms, fill_value=0.5))] += eps
    randoms[torch.isclose(randoms, torch.full_like(randoms, fill_value=-0.5))] -= eps

    mask = torch.where(randoms < 0.5, -1.0, 1.0)
    x_i = icdf(randoms, sigma=sigma) * mask

    f_xi = torch.abs(x_i) * (1.0 / sigma ** 2) * calc_gauss(x_i, mu=0.0, sigma=sigma)
    f_xi[f_xi == 0] += eps
    p_xi = 0.5 * sigma * (2.0 * np.pi)**0.5 * f_xi

    return x_i, p_xi


def smoothFn(func=None, context_args=None, device='cuda'):
    if func is None:
        return functools.partial(smoothFn, context_args=context_args, device=device)

    @functools.wraps(func)
    def wrapper(input_tensor, context_args, *args):
        class SmoothedFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input_tensor, context_args, *args):

                original_input_shape = input_tensor.shape
                importance_fn = importance_gradgauss

                forward_output, avg_img = convolve(grad_of_gaussiankernel, func, importance_fn, input_tensor,
                                                   context_args['nsamples'], context_args, args)

                # save for bw pass
                ctx.fw_out = forward_output
                ctx.original_input_shape = original_input_shape

                return forward_output.mean(), avg_img

            @staticmethod
            def backward(ctx, dy, dz):
                # dz is grad for avg_img

                # Pull saved tensors
                original_input_shape = ctx.original_input_shape
                fw_out = ctx.fw_out
                grad_in_chain = dy * fw_out

                return grad_in_chain.reshape(original_input_shape), None

        return SmoothedFunc.apply(input_tensor, context_args, *args)

    return wrapper
