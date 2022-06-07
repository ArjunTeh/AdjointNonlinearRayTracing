import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import cable
import source
import sensor
import tracer
from utils import plot_utils

from tqdm.auto import tqdm

def run_default_params():
    params = dict(
        hop_distance=3.14,
        cable_length=5,
        cable_radius=1.0,
        cone_ang=30.0,
        camera_span=0.1,
        lr=0.01,
        src_type='planar',
        res_list=[3, 5, 9, 17, 33, 65, 129],
        vol_span=20,
        step_res=2,
        optim_iters=30,
        record_iters=30,
        cone_ang=90,
        nbins=64,
        spp=1,
        npasses=1,
        sensor_distance=1.57
        autodiff=False,
        device='cuda'
    )
    multires_opt(params)


def record_iter(outname, iter_num, n, ngrad, image):

    fig, axes = plt.subplots(2, len(image[0]), squeeze=False)
    for i in range(len(image[0])):
        axes[0, i].imshow(image[0][i])
        axes[0, i].set_title('near')
        axes[1, i].imshow(image[1][i])
        axes[1, i].set_title('far')

    plt.savefig(outname+'/fiber_image_{}.png'.format(iter_num))
    plt.close(fig)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(n)
    ax[0].set_title('radial profile')
    ax[1].plot(ngrad)
    ax[1].set_title('gradient profile')
    plt.savefig(outname+'/fiber_profile_{}.png'.format(iter_num))
    plt.close(fig)



def upres_scene(n, res):

    tween = (n[1:] + n[:-1]) / 2
    nn = torch.zeros((n.shape[0]-1)*2 + 1)
    nn[::2] = n
    nn[1::2] = tween
    nn.requires_grad_(True)

    return nn


def reload_opto(old_o, n, lr):
    # assume that there is only one value to upsample
    ogroup, state = None, None
    for group in old_o.param_groups:
        ogroup = group
        print('beta', ogroup['betas'])
        for p in group['params']:
            if len(old_o.state[p]) == 0:
                # The optimizer hasn't even started yet
                continue
            state = dict()
            ostate = old_o.state[p]
            state['step'] = ostate['step']
            state['exp_avg'] = upres_scene(ostate['exp_avg'], n.shape[0])
            print('range:', ostate['exp_avg'].max(), ostate['exp_avg'].min())
            print('rangeU:', state['exp_avg'].max(), state['exp_avg'].min())
            state['exp_avg_sq'] = upres_scene(ostate['exp_avg_sq'], n.shape[0])

    opto = optim.Adam([n], lr=lr)
    for group in opto.param_groups:
        if ogroup is not None:
            group['betas'] = ogroup['betas']
            group['lr'] = ogroup['lr']
            group['weight_decay'] = ogroup['weight_decay']
            group['eps'] = ogroup['eps']
        for p in group['params']:
            if state is not None:
                opto.state[p] = state
    return opto


def multires_opt(params):
    init_offset = params.get('init_offset', 0)
    outfolder = params.get('outfolder', 'fiber')
    res_list = params.get('res_list', [32])
    cable_length = params.get('cable_length', res_list[-1])
    cable_radius = params.get('cable_radius', res_list[-1])
    camera_span = params.get('camera_span', cable_radius)
    cone_ang = params.get('cone_ang', 100.0)
    src_type = params.get('src_type', 'planar')
    spp = params.get('spp', 1)
    npasses = params.get('npasses', 2)
    sensor_dist = params.get('sensor_distance', 0)
    hop_dist = params.get('hop_distance', 3.14)
    hop_weight = params.get('hop_weight', 0.1)
    run_dir = params.get('run_dir')
    optim_iters = params.get('optim_iters', 300)
    record_iters = params.get('record_iters', optim_iters)
    nbins = params.get('nbins', res_list[-1])
    projected_step = params.get('projected_step', False)
    tdevice = params.get('device', 'cuda')
    lr = params.get('lr', 1e-4)
    autodiff = params.get('autodiff', False)
    plane_eps = params.get('plane_epsilon', 0.001)

    def gen_start_rays(samples=1):
        sdx = sensor_dist - cable_radius*2
        if src_type == 'planar':
            iv = source.plane_source3_rand(torch.tensor([0.0]), (nbins, nbins), spp, cable_radius*2, circle=True, sensor_dist=sdx)
        else:
            iv = source.cone_source3_rand(torch.tensor(0.0), (nbins, nbins), spp, cable_radius*2, sensor_dist=sensor_dist, cone_angle=cone_ang)
        return [x.to(device=tdevice) for x in iv]

    (x, v, planes) = gen_start_rays(spp)
    nrays = x.shape[0]

    def get_sensor_list(planes):
        sensor_p = planes[None, 0, 0, :]
        sensor_n = planes[None, 0, 1, :]
        sensor_t = planes[None, 0, 2, :]
        return sensor_p, sensor_n, sensor_t

    sensor_p, sensor_n, sensor_t = get_sensor_list(planes)

    writer = logging.setup_writer(params, [])
    loss_fn = torch.nn.MSELoss(reduction='sum')

    if autodiff:
        trace_fun = tracer.ADCableTracerC.apply
    else:
        trace_fun = tracer.BackCableTracerC.apply

    def trace(nt, rays, plane):
        x, v = rays
        sp, sn = plane
        sds = cable_radius / nt.shape[0] / 2

        volum = cable.Cable(nt, cable_radius, cable_length)
        n_bound, _ = volum.GetLinear(x)
        v = v / n_bound[:, None]

        xt, vt, dist2 = trace_fun(nt, cable_radius, cable_length, x, v, sp, sds)
        return xt, vt, dist2

    def ground_truth(res):
        return torch.sqrt(2 - torch.linspace(0, 1, res)**2)

    n = torch.ones(res_list[0])
    n += init_offset
    n.requires_grad_(True)
    opto = optim.Adam([n], lr=lr)

    MAX_ITERS_PER_STEP = optim_iters
    cum_steps = 0
    disable_progress = False
    for res_iter in tqdm(range(len(res_list)), disable=disable_progress):

        for j in tqdm(range(MAX_ITERS_PER_STEP*((res_iter+1))), disable=disable_progress):
            opto.zero_grad()

            # TODO(ateh): assumes only one view
            meas_loss = torch.tensor(0, dtype=torch.double)
            loss_0_cum = 0
            loss_1_cum = 0
            near_images = 0
            far_images = []
            n.requires_grad_(True)
            rays_ic = gen_start_rays(spp)
            sensor_p, sensor_n, sensor_t = get_sensor_list(rays_ic[2])

            rays_ic = [r.split(r.shape[0]//npasses) for r in rays_ic]
            end_rays1 = []
            end_rays2 = []
            for i in range(npasses):
                x, v, planes = [r[i] for r in rays_ic]
                sn = planes[:, 1, :]
                sp = planes[:, 0, :]
                xm, vm, dist2 = trace(n, (x, v), (sp, sn))

                eps_mask = dist2 > plane_eps**2
                loss_vec = (xm[eps_mask] - sp[eps_mask])**2 / nrays / cable_radius
                near_loss = torch.sum(loss_vec) / camera_span
                loss_0_cum += near_loss.item()

                near_loss.backward()
                meas_loss += near_loss.item()

                with torch.no_grad():
                    end_rays1.append((xm.detach().clone(), vm.detach().clone()))

                xm, vm, dist2 = trace(n, (x, v), (sp + hop_dist*sn, sn))

                eps_mask = dist2 > plane_eps**2
                loss_vec = (xm[eps_mask] - (sp[eps_mask]+hop_dist*sn[eps_mask]))**2 / nrays / cable_radius
                far_loss = hop_weight * torch.sum(loss_vec) / camera_span
                loss_1_cum += far_loss.item()

                far_loss.backward()
                meas_loss += far_loss.item()
                
                with torch.no_grad():
                    end_rays2.append((xm.detach().clone(), vm.detach().clone()))

                del xm, vm
            with torch.no_grad():
                if (j+cum_steps) % record_iters == 0 or (j+cum_steps) == optim_iters-1:
                    end_rays1 = zip(*end_rays1)
                    end_rays1 = [torch.cat(er) for er in end_rays1]
                    near_images = [sensor.generate_sensor(end_rays1, 1, (sensor_p, sensor_n), nbins, camera_span, sensor_t)]
                    near_images = [source.sum_norm(ni) for ni in near_images]

                    end_rays2 = zip(*end_rays2)
                    end_rays2 = [torch.cat(er) for er in end_rays2]
                    far_images = [sensor.generate_sensor(end_rays2, 1, (sensor_p + hop_dist*sensor_n, sensor_n), nbins, camera_span, sensor_t)]
                    far_images = [source.sum_norm(ni) for ni in far_images]

            if j+cum_steps % record_iters == 0 or j+cum_steps == optim_iters-1:
                record_iter(outfolder, j+cum_steps, n.detach(), n.grad.detach(), (near_images, far_images))

            with torch.no_grad():
                n.grad[-1] = 0

            opto.step()
            if projected_step:
                with torch.no_grad():
                    n.clamp_(min=1)

            del far_images, near_images
            del x, v, planes

        with torch.no_grad():
            inter_state_name = run_dir+'/'+params['exp_name']
            torch.save({
                'rif_state': n,
                'optimizer_state_dict': opto.state_dict(),
                'loss': meas_loss
            }, inter_state_name)
            if res_iter < len(res_list)-1:
                n = upres_scene(n, res_list[res_iter+1])
                # opto = reload_opto(opto, n, (0.5**res_iter)*lr)
                opto = optim.Adam([n], lr=(0.5**res_iter)*lr)
        cum_steps += j

    # get the final output after the optimization
    (x, v, planes) = gen_start_rays(spp*2)
    sp = planes[:, 0, :]
    sn = planes[:, 1, :]
    sensor_p, sensor_n, sensor_t = get_sensor_list(planes)
    xm, vm, dist2 = trace(n, (x, v), (sp, sn))

    images = [sensor.generate_sensor((xm, vm), 1, (sensor_p, sensor_n), nbins, camera_span, sensor_t)]
    images = [source.sum_norm(im) for im in images]

    record_iter(outfolder, cum_steps, n.detach(), n.grad.detach(), (images, images))

    # save results
    torch.save({
        'rif_state': n,
        'final_image': images,
        'optimizer_state_dict': opto.state_dict(),
        'loss': meas_loss
    }, run_dir+'/'+params['exp_name'])

    return n

if __name__ == '__main__':
    run_default_params()
