import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm.auto import tqdm

import grid
import source
import sensor
import tracer
import optimizer
from utils import plot_utils


def multires_opt(params):
    scene = params['scene']
    src_image = params['source_image']
    meas_focal = params['focal_stack']
    meas_dists = params.get('sensor_dists', None)
    res_list = params.get('res_list', [3, 5, 9, 17, 33, 65])
    vol_span = params.get('vol_span', 1)
    spp = params.get('spp', 1)
    sensor_dist = params.get('sensor_distance', 0)
    step_res = params.get('step_res', 2)
    angle_s = params.get('angle_span', 360)
    far_sensor_span = params.get('far_sensor_span', 120)
    nbins = params.get('nbins', scene.shape[0])
    tdevice = params.get('device', 'cuda')
    lr = params.get('lr', 1e-4)
    src_type = params.get('source_type', 'planar')
    autodiff = params.get('autodiff', False)
    optim_iters = params.get("optim_iters", 300)
    record_iters = params.get("record_iters", optim_iters//10 + 1)

    h = vol_span / np.maximum(res_list[-1] - 1, 1)
    ds = h/step_res

    span = vol_span
    measurements = torch.stack(meas_focal)

    def gen_start_rays(samples=1):
        nviews = 1
        if src_type == 'planar':
            iv, rpv = source.rand_rays_in_sphere(nviews, (nbins, nbins), samples, span, angle_span=angle_s, circle=False, xaxis=False, sensor_dist=0)
            tpv = torch.ones(iv[0].shape[0])
        elif src_type == 'point':
            iv, rpv = source.rand_ptrays_in_sphere(nviews, (nbins, nbins), samples, span, angle_span=angle_s, circle=False, xaxis=False, sensor_dist=0)
            tpv = torch.ones(iv[0].shape[0])
        elif src_type == 'cone':
            iv, tpv, rpv = source.rand_cone_in_sphere(nviews, (nbins, nbins), samples, span, angle_span=angle_s, circle=False, xaxis=False, sensor_dist=0, cone_angle=src_angle)
        else:
            iv, _, tpv, rpv = source.rand_area_in_sphere(nviews, (nbins, nbins), samples, span, angle_span=angle_s, circle=False, xaxis=False, sensor_dist=1.0)

        return [x.to(device=tdevice) for x in iv], rpv, tpv

    (x, v, planes), rpv, tpv = gen_start_rays(spp)

    def get_sensor_list(planes, rpv):
        sensor_n = planes[None, 0, 1, :]
        sensor_t = planes[None, 0, 2, :]
        sensor_p = planes[None, 0, 0, :]
        return sensor_p, sensor_n, sensor_t

    loss_fn = torch.nn.MSELoss(reduction='mean')

    if autodiff:
        trace_fun = tracer.ADTracerC.apply
    else:
        trace_fun = tracer.BackTracerC.apply

    def trace(nt, rays):
        x, v = rays
        h = vol_span / np.maximum(nt.shape[0]-1, 1)
        xt, vt = trace_fun(nt, x, v, h, ds)
        return xt, vt

    n = params['init']
    n.requires_grad_(True)

    MAX_ITERS_PER_STEP = optim_iters
    def loss_function(eta):

        meas_loss = torch.tensor(0, dtype=torch.double)
        loss_near_cum, loss_far_cum = 0, 0
        near_images = 0
        far_images = 0
        n.requires_grad_(True)
        rays_ic, rpv, tpv = gen_start_rays(spp)
        sensor_p, sensor_n, sensor_t = get_sensor_list(rays_ic[2], rpv)

        x, v, planes = rays_ic
        with torch.no_grad():
            e = sensor.get_sdf_vals_near((x, v), src_image, (sensor_p - span+meas_dists[0]*sensor_n, sensor_n), span, sensor_t)
        xm, vm = trace(eta, (x, v))

        nim_pass = [sensor.generate_sensor((xm, vm), e, (sp, sn), nbins, span, st)
                    for sp, sn, st in zip(sensor_p, sensor_n, sensor_t)]
        nim_pass = torch.stack([source.sum_norm(ni) for ni in nim_pass])
        loss = loss_fn(nim_pass, measurements)

        del xm, vm
        del far_images, near_images
        del x, v, planes

        return loss

    def log_function(iter_count, eta):
        if iter_count % record_iters == 0 or iter_count == optim_iters-1:
            (x, v, planes), rpv, tpv = gen_start_rays(spp*2)
            sensor_p, sensor_n, sensor_t = get_sensor_list(planes, rpv)
            xm, vm = trace(eta, (x, v))

            e = sensor.get_sdf_vals_near((x, v), src_image, (sensor_p - (span+meas_dists[0])*sensor_n, sensor_n), span, sensor_t)

            images = [sensor.generate_sensor((xm, vm), e, (sensor_p + dist*sensor_n, sensor_n), nbins, span, sensor_t)
                      for dist in meas_dists]
            images = [source.sum_norm(im) for im in images]
            plot_utils.save_multiple_images(images, 'results/multiview/multiview_{}.png'.format(iter_count))

    final_eta, loss_hist = optimizer.multires_opt(loss_function, n, optim_iters, res_list, log_function, lr=lr, statename='results/luneburg/result')

    plt.figure()
    plt.plot(loss_hist)
    plt.savefig('results/multiview/loss_plot.png')
    plt.close()

    return final_eta
