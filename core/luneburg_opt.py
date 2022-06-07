import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import grid
import source
import sensor
import tracer
import optimizer
from utils import plot_utils


def run_default_opt():
    params = dict(
      cube_rots=1,
      res_list=[3, 5, 9, 17, 33, 65, 129],
      vol_span=20,
      step_res=2,
      optim_iters=70,
      record_iters=20,
      angle_span=360,
      nbins=128,
      spp=10,
      planar_source='plane',
      sensor_distance=0
      autodiff: False
      device: 'cuda'
      lr: 0.001
    )
    run_opt(params)


def run_opt(params):
    res_list = params.get('res_list', [3, 5, 9, 17, 33, 65])
    vol_span = params.get('vol_span', res_list[0])
    spp = params.get('spp', 2)
    sensor_dist = params.get('sensor_distance', 0)
    step_res = params.get('step_res', 2)
    optim_iters = params.get('optim_iters', 30)
    record_iters = params.get('record_iters', 30)
    nbins = params.get('nbins', 128)
    tdevice = params.get('device', 'cuda')
    lr = params.get('lr', 1e-2)
    plane_src = params.get('planar_source', 'plane')
    autodiff = params.get('autodiff', False)
    cube_rots = params.get('cube_rots', 1)

    h = vol_span / np.maximum(res_list[-1] - 1, 1)
    ds = h/step_res

    span = vol_span

    def gen_start_rays(samples=1):
        ics = [source.rand_rays_cube((nbins, nbins), samples, span, circle=True, src_type=plane_src) for i in range(cube_rots)]
        ivs, rpv = zip(*ics)
        ivs = [source.random_rotate_ic(*v, span) for v in ivs]
        iv = [torch.cat(v) for v in zip(*ivs)]
        rpv = np.concatenate([np.array(r) for r in rpv])
        return [x.to(device=tdevice) for x in iv], list(rpv)

    def gen_camera_rays(samples=1):
        iv, rpv = source.rand_rays_cube((nbins, nbins), samples, span, circle=True)
        iv = source.random_rotate_ic(*iv, span)
        # iv, rpv = source.rand_rays_in_sphere(nviews, (nbins, nbins), samples, span, angle_span=angle_s, circle=True, xaxis=False)
        return iv, rpv

    (x, v, planes), rpv = gen_start_rays(spp)
    nrays = x.shape[0]

    def get_sensor_list(planes, rpv):
        sensor_n, sensor_p, sensor_t = [], [], []
        offset = 0
        for i in range(len(rpv)):
            sensor_n.append(planes[None, offset, 1, :])
            sensor_t.append(planes[None, offset, 2, :])
            sensor_p.append(planes[None, offset, 0, :] + sensor_dist*sensor_n[-1])
            offset += rpv[i]
        return sensor_p, sensor_n, sensor_t

    if autodiff:
        trace_fun = tracer.ADTracerC.apply
    else:
        trace_fun = tracer.BackTracerC.apply

    def trace(nt, rays):
        x, v = rays
        h = vol_span / np.maximum(nt.shape[0]-1, 1)
        xt, vt = trace_fun(nt, x, v, h, ds)
        return xt, vt

    n = torch.ones(res_list[0], res_list[0], res_list[0]).cuda()

    def loss_function(eta):
        rays_ic, rpv = gen_start_rays(spp)

        x, v, planes = rays_ic
        xm, vm = trace(eta, (x, v))
        sn = planes[:, 1, :]
        sp = planes[:, 0, :] + sensor_dist*sn
        xmp, vmp = sensor.trace_rays_to_plane((xm, vm), (sp, sn))

        near_loss = torch.sum((xmp - sp)**2) / nrays / span

        loss = near_loss
        del xm, vm
        return loss


    def log_function(iter_count, eta):
        if iter_count % record_iters == 0 or iter_count == optim_iters-1:
            rays_ic, rpv = gen_camera_rays(spp)
            sensor_p, sensor_n, sensor_t = get_sensor_list(rays_ic[2], rpv)

            x, v, planes = rays_ic
            xm, vm = trace(eta, (x, v))
            sn = planes[:, 1, :]
            sp = planes[:, 0, :] + sensor_dist*sn
            xmp, vmp = xm.split(rpv), vm.split(rpv)

            near_images = [sensor.generate_sensor((xv, vv), 1, (sp, sn), nbins, span, st)
                           for xv, vv, sp, sn, st in zip(xmp, vmp, sensor_p, sensor_n, sensor_t)]
            near_images = [source.sum_norm(ni) for ni in near_images]
            plot_utils.save_multiple_images(near_images, 'results/luneburg/luneburg_{}.png'.format(iter_count))

    final_eta, loss_hist = optimizer.multires_opt(loss_function, n, optim_iters, res_list, log_function, lr=lr, statename='results/luneburg/result')

    plt.figure()
    plt.plot(loss_hist)
    plt.savefig('results/luneburg/loss_plot.png')
    plt.close()

    return final_eta

if __name__ == '__main__':
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    run_opt(dict())