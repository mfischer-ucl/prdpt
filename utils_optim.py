import time 
import torch 
import utils_fns

from utils_general import show_with_error, plt_errors
from utils_mitsuba import get_mts_rendering, render_smooth


def run_optimization(hparams,
                     optim,
                     theta,
                     gt_theta,
                     ctx_args,
                     schedule_fn,
                     update_fn,
                     plot_initial,
                     plot_interval,
                     plot_intermediate,
                     print_param=False):
    sigma = hparams['sigma']

    reference_image = get_mts_rendering(gt_theta, update_fn, ctx_args)
    initial_image = get_mts_rendering(theta, update_fn, ctx_args)
    ctx_args['gt_image'] = reference_image

    # --------------- set up smoothed renderer
    perturbed_mts = utils_fns.smoothFn(render_smooth,
                                       context_args=None,
                                       device=ctx_args['device'])

    if plot_initial:
        show_with_error(initial_image, reference_image, 0)

    img_errors, param_errors = [], []
    img_errors.append(torch.nn.MSELoss()(initial_image, reference_image).item())
    print(f"Running {hparams['epochs']} epochs with {hparams['nsamples']} samples and sigma={hparams['sigma']}")

    # --------------- run optimization
    for j in range(hparams['epochs']):
        start = time.time()
        optim.zero_grad()

        loss, _ = perturbed_mts(theta.unsqueeze(0), ctx_args)
        loss.backward()

        optim.step()

        # potential sigma scheduling:
        if j > hparams['anneal_const_first'] and hparams['sigma_annealing'] and sigma >= hparams['anneal_sigma_min']:
            sigma = schedule_fn(sigma, curr_iter=j + 1, n=hparams['epochs'],
                                sigma_initial=hparams['sigma'],
                                sigma_min=hparams['anneal_sigma_min'],
                                const_first_n=hparams['anneal_const_first'],
                                const_last_n=hparams['anneal_const_last'])
            ctx_args['sigma'] = sigma
        iter_time = time.time() - start

        # logging, timing, plotting, etc...
        with torch.no_grad():

            # calc loss btwn rendering with current parameter (non-blurred)
            img_curr = get_mts_rendering(theta, update_fn, ctx_args)
            img_errors.append(torch.nn.MSELoss()(img_curr, ctx_args['gt_image']).item())
            param_errors.append(torch.nn.MSELoss()(theta, gt_theta).item())

            # plot intermediate
            if j % plot_interval == 0 and j > 0 and plot_intermediate:

                show_with_error(img_curr, ctx_args['gt_image'], j)

                if len(param_errors) > 1:
                    plt_errors(img_errors, param_errors, title=f'Ep {j+1}')

            pstring = ' - CurrentParam: {}'.format(theta.tolist()) if print_param else ''
            print(f"Iter {j + 1}, ParamLoss: {param_errors[-1]:.6f}, "
                  f"ImageLoss: {img_errors[-1]:.8f} - Time: {iter_time:.4f}{pstring}")

    plt_errors(img_errors, param_errors, title=f'Final, after {hparams["epochs"]} iterations')
    show_with_error(img_curr, ctx_args['gt_image'], hparams['epochs'])
    print("Done.")


