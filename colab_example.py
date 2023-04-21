import torch
import functools
import numpy as np
import matplotlib.pyplot as plt

device = 'cpu'

from utils_fns import smoothFn
from utils_mitsuba import render_smooth
from utils_general import run_scheduler_step, plt_errors


def get_rendering(theta):
    return draw_rect(theta[0], theta[1])


def draw_rect(px, py):
    s, w, h, exp = 256, 0.1, 0.1, 10
    ss = torch.linspace(0, 1, s, device=device)
    x, y = torch.meshgrid(ss, ss)
    image = 1 - 2 * (abs(((py - x) / w)) ** exp + abs((y - px) / h) ** exp)
    return torch.flipud(image.clamp(0, 1))


def plot(curr, ref, blurred_img=None):
    fig, ax = plt.subplots(1, 3 if blurred_img is None else 4)
    mae = torch.abs(curr - ref)
    if blurred_img is None:
        titles, images = ['Initial', 'Reference', 'MAE'], [curr, ref, mae]
    else:
        titles, images = ['Initial', 'Current Blurred', 'Reference', 'MAE'], [curr, blurred_img, ref, mae]
    for j, (t, img) in enumerate(zip(titles, images)):
        ax[j].imshow(img.detach().cpu().numpy())
        ax[j].set_title(t)
        ax[j].axis('off')
    plt.show()


n_samples = 4  # @param {type:"slider", min:1, max:20, step:1}
sigma = 0.06  # @param {type:"slider", min:0.01, max:0.15, step:0.01}

hparams = {'nsamples': n_samples,
           'sigma': sigma,
           'sampler': 'importance',
           'learning_rate': 1e-5,
           'sigma_annealing': True,
           'epochs': 1000,
           'anneal_const_first': 500,
           'anneal_const_last': 0,
           'anneal_sigma_min': 0.05
           }

torch.manual_seed(12)
sigma = hparams['sigma']

# set up initial and gt translation:
theta = torch.tensor([0.5, 0.66], requires_grad=True, device=device)
gt_theta = torch.tensor([0.5, 0.33], device=device)

# set up optim:
optim = torch.optim.Adam([theta], lr=hparams['learning_rate'])

init_img = get_rendering(theta)
ref_img = get_rendering(gt_theta)

# set up ctx_args:
ctx_args = {'antithetic': True, 'nsamples': hparams['nsamples'], 'sigma': hparams['sigma'],
            'sampler': hparams['sampler'], 'device': device, 'gt_image': ref_img}

plot(init_img, ref_img)
plt.rcParams['figure.figsize'] = (12, 6)

# set up smoothed renderer
get_smoothed_loss = smoothFn(render_smooth,
                             context_args=None,
                             device=ctx_args['device'])

img_errors, param_errors = [], []

# run optimization
for j in range(hparams['epochs']):
    optim.zero_grad()

    loss, img_avg = get_smoothed_loss(theta.unsqueeze(0), ctx_args)
    loss.backward()

    optim.step()

    # sigma scheduling:
    if j > hparams['anneal_const_first'] and hparams['sigma_annealing'] and sigma >= hparams['anneal_sigma_min']:
        sigma = run_scheduler_step(sigma, curr_iter=j + 1, sigma_initial=hparams['sigma'],
                                   sigma_min=hparams['anneal_sigma_min'],
                                   n=hparams['epochs'], const_first_n=hparams['anneal_const_first'],
                                   const_last_n=hparams['anneal_const_last'])
        ctx_args['sigma'] = sigma

    # plotting, logging, printing...
    img_curr = get_rendering(theta)
    img_loss = torch.nn.MSELoss()(img_curr, ref_img).item()
    param_loss = torch.nn.MSELoss()(theta, gt_theta).item()
    img_errors.append(img_loss)
    param_errors.append(param_loss)

    if (j + 1) % 25 == 0:
        print(f"Iter {j + 1} - Img.Loss: {img_loss:.4f} - Param.Loss: {param_loss:.4f}")

    if (j + 1) % 10 == 0:
        plot(img_curr, ref_img, img_avg)
        plt_errors(img_errors, param_errors, title=f'Iter {j + 1}')
