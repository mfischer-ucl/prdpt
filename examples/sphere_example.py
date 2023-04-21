import torch
import drjit as dr
import mitsuba as mi

from utils_optim import run_optimization
from utils_general import run_scheduler_step
from utils_mitsuba import setup_shadowscene

if torch.cuda.is_available():
    mi.set_variant('cuda_ad_rgb')


def apply_translation(theta, p, mat_id, init_vpos):
    if isinstance(theta, torch.Tensor):
        theta = theta.tolist()
    trans = mi.Transform4f.translate([0.0, theta[0], theta[1]])
    p[mat_id] = dr.ravel(trans @ init_vpos)
    p.update()


if __name__ == '__main__':

    hparams = {'resx': 256,
               'resy': 192,
               'nsamples': 1,
               'sigma': 0.5,
               'render_spp': 32,
               'initial_translation': [-0.5, 2.5],
               'gt_translation': [-1.5, 1.0],
               'learning_rate': 2e-2,
               'epochs': 400,
               'sigma_annealing': True,
               'anneal_const_first': 200,
               'anneal_const_last': 0,
               'anneal_sigma_min': 0.01,
               'integrator': 'path',
               'max_depth': 6,
               'reparam_max_depth': 2}

    plot_initial = True
    plot_intermediate = False
    plot_interval = 100

    device = 'cuda'
    torch.manual_seed(0)
    update_fn = apply_translation

    # --------------- set up initial and gt translation:
    initial_translation = torch.tensor(hparams['initial_translation'], requires_grad=True, device=device)
    gt_translation = torch.tensor(hparams['gt_translation'], device=device)

    # --------------- set up optimizer:
    optim = torch.optim.Adam([initial_translation], lr=hparams['learning_rate'])

    # --------------- set up scene:
    scene, params, mat_id, initial_vertex_positions = setup_shadowscene(hparams)
    dr.disable_grad(params)

    # --------------- set up ctx_args
    ctx_args = {'scene': scene, 'params': params, 'spp': hparams['render_spp'],                     # rendering
                'init_vpos': initial_vertex_positions, 'mat_id': mat_id, 'update_fn': update_fn,    # rendering
                'sampler': 'importance', 'antithetic': True, 'nsamples': hparams['nsamples'],       # ours
                'sigma': hparams['sigma'], 'device': device}                                        # ours

    run_optimization(hparams=hparams,
                     optim=optim,
                     theta=initial_translation,
                     gt_theta=gt_translation,
                     ctx_args=ctx_args,
                     schedule_fn=run_scheduler_step,
                     update_fn=apply_translation,
                     plot_initial=plot_initial,
                     plot_interval=plot_interval,
                     plot_intermediate=plot_intermediate)
