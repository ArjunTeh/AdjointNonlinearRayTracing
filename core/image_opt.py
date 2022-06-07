import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm.auto import tqdm
from PIL import Image

import grid
import source
import sensor
import tracer
import optimizer
from utils import plot_utils


def multires_opt(params, result_dir):
    disp_ims = params.get('disp_ims', [None])
    defl_ims = params.get('defl_ims', [None])
    defl_weight = params.get('defl_weight', 1.0)
    sdf_loss = params.get('sdf_loss', False)
    sdf_disp = params.get('sdf_disp', [None])
    sdf_defl = params.get('sdf_defl', [None])
    res_list = params.get('res_list', [3, 5, 9, 17, 33, 65])
    vol_span = params.get('vol_span', 1)
    spp = params.get('spp', 1)
    sensor_dist = params.get('sensor_distance', 0)
    step_res = params.get('step_res', 2)
    angle_s = params.get('angle_span', 360)
    far_sensor_span = params.get('far_sensor_span', 120)
    nbins = params.get('nbins', 128)
    tdevice = params.get('device', 'cuda')
    lr = params.get('lr', 1e-4)
    src_type = params.get('source_type', 'planar')
    autodiff = params.get('autodiff', False)
    optim_iters = params.get("optim_iters", 300)
    record_iters = params.get("record_iters", optim_iters//10 + 1)

    h = vol_span / np.maximum(res_list[-1] - 1, 1)
    ds = h/step_res

    span = vol_span
    nviews = max(len(disp_ims), len(defl_ims))

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

    (x, v, planes), rpv, tpv = gen_start_rays(spp)

    def get_sensor_list(planes, rpv):
        sensor_n, sensor_p, sensor_t = [], [], []
        offset = 0
        for i in range(nviews):
            sensor_n.append(planes[None, offset, 1, :])
            sensor_t.append(planes[None, offset, 2, :])
            sensor_p.append(planes[None, offset, 0, :])# + sensor_dist*sensor_n[-1])
            offset += rpv[i]
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

    n = params.get('init', torch.ones((res_list[0],)*3))

    MAX_ITERS_PER_STEP = optim_iters
    def loss_function(eta):

        meas_loss = torch.tensor(0, dtype=torch.double)
        near_images = 0
        far_images = 0
        n.requires_grad_(True)
        rays_ic, rpv, tpv = gen_start_rays(spp)
        sensor_p, sensor_n, sensor_t = get_sensor_list(rays_ic[2], rpv)

        x, v, planes = rays_ic
        xm, vm = trace(eta, (x, v))
        sn = planes[:, 1, :]
        sp = planes[:, 0, :]
        xmp, vmp = sensor.trace_rays_to_plane((xm, vm), (sp, sn))

        xm_s, vm_s = xmp.split(rpv), vmp.split(rpv)
        dists = (1/(tpv**2)).split(rpv)

        near_loss = 0
        near_images = []
        near_images = [sensor.generate_sensor((xv, vv), d, (sp, sn), nbins, span, st)
                       for xv, vv, sp, sn, st, d in zip(xm_s, vm_s, sensor_p, sensor_n, sensor_t, dists)]
        near_images = [source.sum_norm(ni) for ni in near_images]
        if sdf_loss and (sdf_disp[0] is not None):
            near_sdf = [sensor.get_sdf_vals_near((xv, vv), sdi, (sp, sn), span, st)
                        for xv, vv, sdi, sp, sn, st in zip(xm_s, vm_s, sdf_disp, sensor_p, sensor_n, sensor_t)]
            near_loss = sum([(sdi**2).sum() / sdi.numel() for sdi in near_sdf])
        elif disp_ims[0] is not None:
            near_loss = sum([loss_fn(im, meas) for im, meas in zip(near_images, disp_ims)]) / len(disp_ims)

        far_loss = 0
        far_images = []
        far_images = [sensor.generate_inf_sensor((xv, vv), 1, (sp, sn), nbins, far_sensor_span, st)
                      for xv, vv, sp, sn, st in zip(xm_s, vm_s, sensor_p, sensor_n, sensor_t)]
        far_images = [source.sum_norm(fi) for fi in far_images]
        if sdf_loss and (sdf_defl[0] is not None):
            far_sdf = [sensor.get_sdf_vals_far((xv, vv), sdi, (sp, sn), far_sensor_span, st)
                       for xv, vv, sdi, sp, sn, st in zip(xm_s, vm_s, sdf_defl, sensor_p, sensor_n, sensor_t)]
            far_loss = defl_weight * sum([(sdi**2).sum() / sdi.numel() for sdi in far_sdf])
        elif defl_ims[0] is not None:
            far_loss = defl_weight * sum([loss_fn(im, meas) for im, meas in zip(far_images, defl_ims)])

        loss = near_loss + far_loss
        meas_loss += loss.item()

        del xm, vm
        del far_images, near_images
        del x, v, planes

        return loss

    def log_function(iter_count, eta):
        if iter_count % record_iters == 0 or iter_count == optim_iters-1:
            (x, v, planes), rpv, tpv = gen_start_rays(spp*2)
            sensor_p, sensor_n, sensor_t = get_sensor_list(planes, rpv)
            xm, vm = trace(eta, (x, v))
            xm_s, vm_s = xm.split(rpv), vm.split(rpv)
            dists = (1/(tpv**2)).split(rpv)

            images = [sensor.generate_sensor((xv, vv), d, (sp, sn), nbins, span, st)
                      for xv, vv, sp, sn, st, d in zip(xm_s, vm_s, sensor_p, sensor_n, sensor_t, dists)]
            images = [source.sum_norm(im) for im in images]
            plot_utils.save_multiple_images(images, result_dir+'/multiview_{}.png'.format(iter_count))

    final_eta, loss_hist = optimizer.multires_opt(loss_function, n, optim_iters, res_list, log_function, lr=lr, statename='results/luneburg/result')

    plt.figure()
    plt.plot(loss_hist)
    plt.savefig(result_dir+'/loss_plot.png')
    plt.close()

    return final_eta

def run_multiview_exp():
    resolution = 128
    einstein_im = Image.open("data/einstein.png").resize((resolution, resolution))
    einstein_im = torch.from_numpy(np.asarray(einstein_im).astype(np.float32)).cuda()
    turing_im = Image.open("data/turing.png").resize((resolution, resolution))
    turing_im = torch.from_numpy(np.asarray(turing_im).astype(np.float32)).cuda()

    disp_images = [
        source.sum_norm(einstein_im),
        source.sum_norm(turing_im)
    ]
    params = dict(
        disp_ims=disp_images,
        optim_iters=10,
        record_iters=10
    )

    multires_opt(params, 'results/multiview')


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    run_multiview_exp()
