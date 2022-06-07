import torch
import numpy as np
import path_matrix
from core import source
import scipy.sparse as sparse

from utils import voxel_scenes
from core import tracer as ekTracer

import time
import matplotlib.pyplot as plt


def inout_cube(x, span):
    f = torch.all(x <= (span+1e-6), dim=-1)
    b = torch.all(x >= -1e-6, dim=-1)
    return torch.logical_and(f, b)


def trace_to_cube(sp, sd, span):
    tmin = - sp / sd
    tmax = (span - sp) / sd

    tmin[~torch.isfinite(tmin)] = float('inf')
    tmax[~torch.isfinite(tmax)] = float('inf')

    tmin[tmin < 0] = float('inf')
    tmax[tmax < 0] = float('inf')

    tms, _ = torch.sort(torch.cat([tmin, tmax], dim=-1))

    still = torch.ones(sp.shape[0]).to(bool)
    xp = sp + tms[:, 0, None]*sd
    for i in range(tms.shape[1]):
        npos = sp + tms[:, i, None]*sd
        done = inout_cube(npos, span)

        xp[still & done] = npos[still & done]
        still[done] = 0

        if torch.all(~still):
            break

    if i == (tms.shape[1] - 1):
        print('failed to instersect all rays')

    return xp, sd


def trace_back_to_cube(xp, xd, span):
    tmin = xp / xd
    tmax = - (span - xp) / xd

    tmin[~torch.isfinite(tmin)] = float('-inf')
    tmax[~torch.isfinite(tmax)] = float('-inf')

    tmin[tmin > 0] = float('-inf')
    tmax[tmax > 0] = float('-inf')

    tmin = torch.max(tmin, dim=1)[0]
    tmax = torch.max(tmax, dim=1)[0]
    t = torch.maximum(tmin, tmax)

    return xp + t[:, None]*xd, xd


def plot_slices(rif):
    res_m = rif.shape[0]//2
    plt.subplot(1, 3, 1)
    plt.imshow(rif[:, :, res_m], vmin=rif.min(), vmax=rif.max())
    plt.subplot(1, 3, 2)
    plt.imshow(rif[:, res_m, :], vmin=rif.min(), vmax=rif.max())
    plt.subplot(1, 3, 3)
    plt.imshow(rif[res_m, :, :], vmin=rif.min(), vmax=rif.max())
    plt.colorbar()
    plt.show()


def fuel_reconstruction(scale, plot_vol=False):
    nviews = 32
    nres = 64
    solve_res = 64
    nbins = 64
    spp = 16
    span = 10
    h = span / nres

    bnd = 1 + scale

    torch.manual_seed(0)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    (x, v, p), rpv = source.rand_rays_in_sphere(nviews, (nbins, nbins), spp, 0.9*span, angle_span=180, sensor_dist=1.4*span)
    # (x, v, p), rpv = utils.rand_ptrays_in_sphere(nviews, (nbins, nbins), 1, 0.9*span, angle_span=180, sensor_dist=1.4*span)
    # (x, v, p), d, rpv = utils.rand_cone_in_sphere(nviews, (nbins, nbins), spp, 0.5*span, angle_span=180.0, sensor_dist=1.4*span, cone_angle=5.0)
    x += 0.05*span
    # x += 0.25*span
    x = x - span*(0.5)*v

    # x = x[nbins*nbins*spp:]
    # v = v[nbins*nbins*spp:]

    gtruth = voxel_scenes.load_fuel_injection()
    gtruth = (-scale * gtruth) + bnd
    gtruth = voxel_scenes.to_torch(gtruth.astype(np.float32)).to('cuda')
    # gtruth = gtruth.permute(2, 1, 0)
    tmp = torch.ones(65, 65, 65) * bnd
    tmp[:-1, :-1, :-1] = gtruth
    gtruth = tmp.to('cuda')

    trace_ad = ekTracer.BackTracerC.apply
    xt, vt = trace_ad(gtruth, x, v, h, h/2)
    xt += 2*h*vt
    # x = x - span*v

    # x -= h/2
    # xt -= h/2

    sp, sd = trace_to_cube(x, v, span)
    xp, _ = trace_to_cube(xt, -vt, span)
    # xp += h*vt/4
    xd = vt

    # xp, xd = xt, vt

    dist_d = torch.linalg.norm(xp-sp, dim=1)
    mask = dist_d > 1.74*span
    if torch.any(mask):
        print('x', x[mask][0])
        print('v', v[mask][0])
        print('sp', sp[mask][0])
        print('xp', xp[mask][0])
        print('xt', xt[mask][0])
        print('vt', vt[mask][0])
        raise ValueError("Trace is bad!")

    if torch.any(~torch.isfinite(xt)):
        raise ValueError("Bad Vals!")

    sp = sp.cpu().numpy()
    sd = sd.cpu().numpy()
    xp = xp.cpu().numpy()
    xd = xd.cpu().numpy()
    gtruth = gtruth.cpu().numpy()

    # print(xt)
    # print(xp)
    # print(vt)
    # print(xd)
    # recon = np.load('fuel_reconstruct_65_atcheson_cg.npy')

    phi, _ = path_matrix.construct_voxel_matrix(sp, sd, xp, xd, span, solve_res, 'linear', int_res=nres*4)
    diff_mats = path_matrix.construct_diff_matrices(solve_res, span, 3)
    defl_mat = path_matrix.construct_deflection_matrix_direct(phi, solve_res, span, x.shape[1])
    c_mat, c_sol = path_matrix.construct_boundary_conditions(solve_res, x.shape[1], bnd)

    # grads = [dm.dot(gtruth.flatten(order='F')) for dm in diff_mats]
    # grads_re = [g.reshape(gtruth.shape, order='F') for g in grads]
    # plot_slices(grads_re[0])
    # plot_slices(grads_re[1])
    # plot_slices(grads_re[2])
    # return

    # print(c_mat.shape)
    # print(diff_mats[0].shape)
    # print(diff_mats[1].shape)
    # print(diff_mats[2].shape)

    # dei = 4
    # print('x', x[dei])
    # print('xt', xt[dei])
    # print('sp', sp[dei])
    # print('xp', xp[dei])
    # print('v', v[dei])
    # print('vt', vt[dei])
    # print(phi.toarray())
    # return

    # full_A = sparse.vstack([c_mat, *diff_mats])
    # full_b = c_sol
    # for i in range(len(diff_mats)):
    #     full_b = np.vstack([full_b, grads[i][:, None]])

    # print(full_A.shape)
    # print(full_b.shape)
    # grad_int = sparse.linalg.lsqr(full_A, full_b, show=True)
    # grad_int_r = np.reshape(grad_int[0], gtruth.shape, order='F')
    # plot_slices(grad_int_r)
    # return

    # plt.imshow(grads[0][:, :, 32])
    # plt.show()
    # return

    begin_time = time.time()
    print("PHI SOLVE")
    rif_grad = path_matrix.deflection_solve_gradient(phi, xd-sd, damp=0.000)
    phi_time = time.time() - begin_time

    # plot_slices(rif_grad[0][0].reshape((solve_res,)*3, order='F'))
    # plot_slices(rif_grad[1][0].reshape((solve_res,)*3, order='F'))
    # plot_slices(rif_grad[2][0].reshape((solve_res,)*3, order='F'))

    np.save('fuel_grad_x', rif_grad[0][0].reshape((solve_res,)*3, order='F'))
    np.save('fuel_grad_y', rif_grad[1][0].reshape((solve_res,)*3, order='F'))
    np.save('fuel_grad_z', rif_grad[2][0].reshape((solve_res,)*3, order='F'))

    # rif_grad = [[rg[0].reshape(gtruth.shape, order='C').flatten(order='F')] for rg in rif_grad]
    print("INTEGRATION STEP")
    begin_time = time.time()
    rif_d = path_matrix.gradient_integration(diff_mats, (c_mat, c_sol), rif_grad, damp=0.0001)
    # rif_d = path_matrix.deflection_solve(defl_mat,
    #                                      (c_mat, c_sol),
    #                                      xd-sd,
    #                                      0.01)
    int_time = time.time() - begin_time

    print(rif_d)
    rif_d0 = np.reshape(rif_d[0], (solve_res,)*3, order='F')

    np.save('fuelrecon_64_32v_atcheson_lsqr_'+str(scale), rif_d0)
    if plot_vol:
        plot_slices(rif_d0)

    print("Residual --------------")
    print('phi residual x:', rif_grad[0][3])
    print('phi residual y:', rif_grad[1][3])
    print('phi residual z:', rif_grad[2][3])
    print('grad residual:', rif_d[3])

    print("Error -------")
    error = (rif_d0-gtruth[:-1, :-1, :-1]) / gtruth[:-1, :-1, :-1]
    print('norm rel error:', np.linalg.norm(error))
    print('max rel error:', error.max())
    print('l1 error', np.mean(error))
    print("TIME VALS (s) ------------")
    print("phi solve: {}".format(phi_time))
    print("int solve: {}".format(int_time))
    print("tot solve: {}".format(int_time + phi_time))


def load_fuel_grad():
    gradx = np.load('fuel_grad_x.npy')
    grady = np.load('fuel_grad_y.npy')
    gradz = np.load('fuel_grad_z.npy')

    plot_slices(gradx.flatten().reshape(gradx.shape, order='F'))
    plot_slices(grady.flatten().reshape(grady.shape, order='F'))
    plot_slices(gradz.flatten().reshape(gradz.shape, order='F'))


def run_recon_profile(val):
    import os, psutil
    process = psutil.Process(os.getpid())
    begin = time.time()
    fuel_reconstruction(val)
    end = time.time()

    print('Total Program Time:', end-begin)

    print('MEMORY INFO (MB) -------------')
    print('physical:', process.memory_info().rss / (1024**2))
    print('virtual:', process.memory_info().vms / (1024**2))


if __name__ == '__main__':
    # load_fuel_grad()

    # run_recon_profile(0.0003)
    # run_recon_profile(0.003)
    run_recon_profile(0.3)

    # val1 = np.load('fuelrecon_64_atcheson_lsqr_0.003.npy')
    # val2 = np.load('fuelrecon_64_atcheson_lsqr_0.03.npy')
    # val3 = np.load('fuelrecon_64_atcheson_lsqr_0.3.npy')
    # plot_slices(np.load('fuelrecon_64_atcheson_lsqr_0.0003.npy'))
    # val1 = np.load('fuelrecon_64_32v_atcheson_lsqr_0.0003.npy')
    # val2 = np.load('fuelrecon_64_32v_atcheson_lsqr_0.003.npy')
    # val3 = np.load('fuelrecon_64_32v_atcheson_lsqr_0.03.npy')
    # vals = [val1, val2, val3]
    
    # fig, ax = plt.subplots(3, 3)
    # for i in range(3):
    #     v = vals[i]
    #     ax[0, i].imshow(v[:, :, v.shape[2]//2], vmin=v.min(), vmax=v.max())
    #     ax[1, i].imshow(v[:, v.shape[1]//2, :], vmin=v.min(), vmax=v.max())
    #     ax[2, i].imshow(v[v.shape[0]//2, :, :], vmin=v.min(), vmax=v.max())
    
    # plt.show()