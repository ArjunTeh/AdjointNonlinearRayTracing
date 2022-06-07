import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

import grid
import source
import sensor
import optimizer
import tracer
from utils import plot_utils


def multires_opt(params, result_dir):
    res_list = params.get('res_list', [3, 5, 9, 17, 33, 65])
    vol_span = params.get('vol_span', 1.0)
    spp = params.get('spp', 1)
    nviews = params.get('nviews', 1)
    sensor_dist = params.get('sensor_distance', 0)
    step_res = params.get('step_res', 2)
    optim_iters = params.get('optim_iters', 300)
    record_iters = params.get('record_iters', 30)
    nviews = params.get('nviews', 1)
    angle_s = params.get('angle_span', 360)
    nbins = params.get('nbins', 128)
    tdevice = params.get('device', 'cuda')
    lr = params.get('lr', 1e-4)
    src_type = params.get('source_type', 'planar')
    autodiff = params.get('autodiff', False)
    fuel_val = params.get('fuel_val', 0.0003)
    defl_weight = params.get('defl_weight', 1.0)

    h = vol_span / np.maximum(res_list[-1] - 1, 1)
    ds = h/step_res
    span = vol_span

    # TODO: import fuel injection dataset
    gtruth = voxel_scenes.load_fuel_injection()
    gtruth = (-fuel_val * gtruth) + (1+fuel_val)
    gtruth = voxel_scenes.to_torch(gtruth.astype(np.float32)).to('cuda')
    tmp = torch.ones(65, 65, 65) * (1+fuel_val)
    tmp[:-1, :-1, :-1] = gtruth
    gtruth = tmp


    def gen_start_rays(samples=1):
        if src_type == 'planar':
            iv, rpv = source.rand_rays_in_sphere(nviews, (nbins, nbins), samples, span, angle_span=angle_s, circle=False, xaxis=False, sensor_dist=sensor_dist)
            tpv = torch.ones(iv[0].shape[0])
        elif src_type == 'point':
            iv, rpv = source.rand_ptrays_in_sphere(nviews, (nbins, nbins), samples, span, angle_span=angle_s, circle=False, xaxis=False, sensor_dist=sensor_dist)
            tpv = torch.ones(iv[0].shape[0])
        else:
            iv, _, tpv, rpv = source.rand_area_in_sphere(nviews, (nbins, nbins), samples, span, angle_span=angle_s, circle=False, xaxis=False, sensor_dist=sensor_dist)

        return [x.to(device=tdevice) for x in iv], rpv, tpv

    (xic, vic, planesic), rpv, tpv = gen_start_rays(spp)

    def get_sensor_list(planes, rpv):
        sensor_n, sensor_p, sensor_t = [], [], []
        offset = 0
        for i in range(nviews):
            sensor_n.append(planes[None, offset, 1, :])
            sensor_t.append(planes[None, offset, 2, :])
            sensor_p.append(planes[None, offset, 0, :])# + sensor_dist*sensor_n[-1])
            offset += rpv[i]
        return sensor_p, sensor_n, sensor_t

    sensor_p, sensor_n, sensor_t = get_sensor_list(planesic, rpv)

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

    x_gt, v_gt = trace(gtruth, (xic, vic))
    x_gt, v_gt = sensor.trace_rays_to_plane((x_gt, v_gt), (planesic[:, 0, :], planesic[:, 1, :]))

    n = torch.ones(res_list[0], res_list[0], res_list[0]) + fuel_val

    MAX_ITERS_PER_STEP = optim_iters
    cum_steps = 0
    disable_progress = False

    def loss_function(eta):
        rays_ic = xic, vic, planesic

        x, v, planes = rays_ic
        xm, vm = trace(eta, (x, v))
        sn = planes[:, 1, :]
        sp = planes[:, 0, :]
        xmp, vmp = sensor.trace_rays_to_plane((xm, vm), (sp, sn))

        disp_loss = loss_fn(xmp, x_gt)
        defl_loss = loss_fn(vmp, v_gt)
        loss = (disp_loss + defl_weight*defl_loss) / fuel_val

        del xm, vm
        del x, v, planes

        return loss

    def log_function(iter_count, eta):
        if iter_count % record_iters == 0 or iter_count == optim_iters-1:
            imx = eta[eta.shape[0]//2, :, :]
            imy = eta[:, eta.shape[1]//2, :]
            imz = eta[:, :, eta.shape[2]//2]
            plot_utils.save_multiple_images([imx, imy, imz], result_dir+'/fuel_injection_{}.png'.format(iter_count))

    final_eta, loss_hist = optimizer.multires_opt(loss_function, n, optim_iters, res_list, log_function, lr=lr, statename='results/fuel_injection/result')
    
    plt.figure()
    plt.plot(loss_hist)
    plt.savefig(result_dir+'/loss_plot.png')
    plt.close()

    return final_eta
