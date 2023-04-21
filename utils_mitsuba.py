import os
import torch
import drjit as dr
import mitsuba as mi


def get_mts_rendering(theta, update_fn, ctx_args):
    # returns a rendering with the current parameters.
    # update_fn is a function pointer to the function that updates the scene parameters, e.g., apply_translation
    update_fn(theta, p=ctx_args['params'], init_vpos=ctx_args['init_vpos'], mat_id=ctx_args['mat_id'])
    rendering = mi.render(ctx_args['scene'], ctx_args['params'], seed=0, spp=ctx_args['spp'])
    return torch.tensor(rendering, dtype=torch.float32, device=ctx_args['device'])


def render_smooth(perturbed_theta, update_fn, ctx_args):
    # render with each perturbed position, get the final image, compute loss, batch, return
    # perturbed_thetas is expected to be of dim [nsamples, ndim]
    with torch.no_grad():
        imgs, losses = [], []
        for j in range(perturbed_theta.shape[0]):       # for each sample
            perturbed_img = get_mts_rendering(perturbed_theta[j, :], update_fn, ctx_args)
            perturbed_loss = torch.nn.MSELoss()(perturbed_img, ctx_args['gt_image'])
            imgs.append(perturbed_img)
            losses.append(perturbed_loss)

        # avg_img just for visualization, simple averaging w/o weighting
        avg_img = torch.mean(torch.cat([x.unsqueeze(0) for x in imgs], dim=0), dim=0)
        loss = torch.stack(losses)
    return loss, avg_img


def create_scene_from_xml(xmlpath, resx=512, resy=512, integrator='path', maxdepth=6, reparam_max_depth=2):
    # read a xml scene file w/ generic attributes for integrator and res/spp, convert to specified params, return
    print(os.getcwd())
    lines = open(xmlpath, 'r').readlines()
    for idx in range(len(lines)):
        line = lines[idx]
        if 'resx' in lines[idx]:
            lines[idx] = line.replace('resolution_x', str(resx))
        if 'resy' in lines[idx]:
            lines[idx] = line.replace('resolution_y', str(resy))
        if 'integrator' in lines[idx]:
            lines[idx] = line.replace('integrator_type', integrator)
        if 'max_depth' in lines[idx]:
            if integrator == 'direct':
                lines[idx] = ''
            else:
                lines[idx] = line.replace('depth_value', str(maxdepth))
        if 'reparam_max_depth' in lines[idx]:
            if integrator == 'prb_reparam':
                lines[idx] = line.replace('reparam_depth_value', str(reparam_max_depth))
            else:
                lines[idx] = ''

    tmppath = os.path.join(os.path.split(xmlpath)[0], 'tmp.xml')
    open(tmppath, 'w').writelines(lines)
    scene = mi.load_file(tmppath)
    os.remove(tmppath)
    return scene


def setup_shadowscene(hparams):
    xmlpath = '../scenes/shadows/shadows.xml'
    scene = create_scene_from_xml(xmlpath, resx=hparams['resx'], resy=hparams['resy'], integrator=hparams['integrator'],
                                  maxdepth=hparams['max_depth'], reparam_max_depth=hparams['reparam_max_depth'])
    params = mi.traverse(scene)
    mat_id = 'PLYMesh_1.vertex_positions'       # this changes per scene, adapt accordingly
    initial_vertex_positions = dr.unravel(mi.Point3f, params[mat_id])
    return scene, params, mat_id, initial_vertex_positions

