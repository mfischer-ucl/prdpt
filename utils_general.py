import torch
import matplotlib.pyplot as plt


def update_sigma_linear(it, sigma_0, sigma_min, n=400, const_first=100):
    return sigma_0 - (it - const_first) * (sigma_0 - sigma_min) / (n - const_first)


def run_scheduler_step(curr_sigma, curr_iter, sigma_initial, sigma_min, n, const_first_n, const_last_n=None):
    n_real = n - const_last_n if const_last_n else n
    newsigma = update_sigma_linear(curr_iter, sigma_initial, sigma_min, n_real, const_first_n)
    return newsigma


def show_with_error(init_img, ref_img, iter, suptitle=None):
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(init_img.detach().cpu() ** .4545)
    ax[0].set_title('Init' if iter == 0 else 'Current')
    ax[1].imshow(ref_img.detach().cpu() ** .4545)
    ax[1].set_title('Reference')
    ax[2].imshow(torch.abs((init_img.cpu() - ref_img.cpu())).mean(dim=-1).detach())
    ax[2].set_title('MAE')
    [a.axis('off') for a in ax]
    plt.tight_layout()
    if suptitle: plt.suptitle(suptitle)
    plt.show()

# def show_with_error(init_img, ref_img, potential_nonblurry_img=None, suptitle=None):
#     fig, ax = plt.subplots(1, 3 if potential_nonblurry_img is None else 4)
#     ax[0].imshow(init_img.detach().cpu() ** .4545)
#     ax[0].set_title('Current / Init')
#     if potential_nonblurry_img is not None:
#         ax[1].imshow(potential_nonblurry_img.detach().cpu() ** .4545)
#         ax[1].set_title('Current Non-Blurred')
#     ax[-2].imshow(ref_img.detach().cpu() ** .4545)
#     ax[-2].set_title('Reference')
#     comp_img = init_img if potential_nonblurry_img is None else potential_nonblurry_img
#     ax[-1].imshow(torch.abs((comp_img.cpu() - ref_img.cpu())).mean(dim=-1).detach())
#     ax[-1].set_title('MAE')
#     [a.axis('off') for a in ax]
#     plt.tight_layout()
#     if suptitle: plt.suptitle(suptitle)
#     plt.show()


def plt_errors(img_err, param_err, title):
    imax, pmax = max(img_err), max(param_err)
    plt.plot([x / pmax for x in param_err], c='blue', label='Param. MSE (norm.)')
    plt.plot([x / imax for x in img_err], c='orange', label='Img. MSE (norm.)')
    plt.title(title)
    plt.legend()
    plt.show()