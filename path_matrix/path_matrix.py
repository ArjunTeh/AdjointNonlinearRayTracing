import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, spdiags, vstack
from scipy.sparse import linalg
from scipy.sparse.linalg import LinearOperator
from torch._C import Value

def construct_voxel_matrix(spos, sdir, epos, edir, dim, res, spline='linear', int_res=175, ray_id=0, path=None):
    if spos.size == 0:
        return np.zeros_like(spos), np.zeros_like(spos)
    num_rays = spos.shape[0]
    dimension = spos.shape[1]
    if spline == 'linear':
        spline_func = get_linear_path
    elif spline == 'hermite':
        spline_func = get_hermite_path
    elif spline == 'true':
        spline_func = lambda p0, d0, p1, d1, t : get_true_path(p0, d0, p1, d1, t, path[0], path[1], path[2])

    # get the vector of voxels
    box_dim = dim/np.maximum(1, res)
    phi_data = np.array([])
    phi_row = np.array([])
    phi_col = np.array([])

    # debugging line, display the spline
    spline_data = np.zeros((int_res+1, spos.shape[1]))

    # multiprocessing - for rays
    p_pre = spline_func(spos, sdir, epos, edir, 0)
    ind_pre = which_voxel(p_pre, box_dim, res)
    dist = np.zeros((spos.shape[0],))
    spline_data[0, :] = p_pre[ray_id]
    for j in range(int_res):
        # TODO(ateh): arc length parameterization look up
        # p_pre = spline_func(spos, sdir, epos, edir, j/int_res)
        p_cur = spline_func(spos, sdir, epos, edir, (j+1)/int_res)
        # ind_pre = which_voxel(p_pre, box_dim, res)
        ind_cur = which_voxel(p_cur, box_dim, res)

        idx = ind_pre != ind_cur
        # try:
        #     p_cur[idx] = intersect_line(p_pre[idx], p_cur[idx], ind_pre[idx], ind_cur[idx], box_dim, res)
        # except ValueError:
        #     print('spos', spos[idx][0])
        #     print('svel', sdir[idx][0])
        #     print('epos', epos[idx][0])
        #     raise

        if (j == int_res-1):
            idx = ind_pre == ind_pre

        dist = dist + np.sqrt(((p_cur - p_pre)**2).sum(1))
        # print('iter', j)
        # print(p_pre)
        # print(p_cur)
        # print(dist)
        # print(ind_pre)

        phi_data = np.concatenate([phi_data, dist[idx]])
        phi_col = np.concatenate([phi_col, ind_pre[idx]])
        new_rows = np.array(np.flatnonzero(idx))
        phi_row = np.concatenate([phi_row, new_rows])

        dist[idx] = 0
        ind_pre = ind_cur.copy()
        p_pre[:] = p_cur[:]
        spline_data[j+1, :] = p_pre[ray_id]

    phi_data = np.array(phi_data);
    phi_row = np.array(phi_row);
    phi_col = np.array(phi_col).squeeze();
    phi = coo_matrix((phi_data, (phi_row, phi_col)),
                         shape=(num_rays, res**dimension)).tocsr()

    return phi, spline_data

#TODO: check this function here! The phi*diff seems to be weird!
def construct_diff_matrices(res, vol_dim, dimension):
    num_voxels = res
    box_dim = vol_dim/np.maximum(1, res)

    diff_list = []
    # create a sparse matrix for each dimension
    data = np.concatenate((
        # np.zeros((1, num_voxels)),
        -np.ones((1, num_voxels)),
        np.ones((1, num_voxels)))
                           )
    # data[0, -1] = -1
    # data[1, -1] = 1
    data[0, -1] = 0
    diff = spdiags(data, np.array([0, 1]), num_voxels, num_voxels)
    I = sp.eye(res)

    # construct the diff matrix
    if dimension == 2:
        diff_list.append(sp.kron(I, diff))
        diff_list.append(sp.kron(diff, I))
    elif dimension == 3:
        diff_list.append(sp.kron(I, sp.kron(I, diff)))
        diff_list.append(sp.kron(I, sp.kron(diff, I)))
        diff_list.append(sp.kron(diff, sp.kron(I, I)))

    return [(1/box_dim)*diff for diff in diff_list]
    # return diff_list

def construct_deflection_matrix(phi, diff_mats):
    full_A = None
    for i in range(len(diff_mats)):
        solve = phi.dot(diff_mats[i])
        full_A = vstack([full_A, solve])
    return full_A

def construct_deflection_matrix_direct(phi, res, vol_dim, dimension):
    return construct_deflection_matrix(phi,
                                       construct_diff_matrices(res, vol_dim, dimension))


def construct_boundary_conditions(res, dimension, val):
    num_voxels = res**dimension
    if dimension == 2:
        num_constraints = 4*(res - 1)
    else:
        num_constraints = 6*res*res - 12*res + 8

    row, col, data = (np.zeros(num_constraints) for i in range(3))

    idx = 0
    for i in range(num_voxels):
        z = i // (res*res)
        y = (i % (res*res)) // res
        x = i % res
        if x == 0 or y == 0 or (z == 0 and dimension>2) or \
           x == (res-1) or y==(res-1) or z==(res-1):
            row[idx] = idx
            col[idx] = i
            data[idx] = 1
            idx += 1

    c_mat = coo_matrix((data, (row, col)), shape=(num_constraints, num_voxels))
    c_sol = val * np.ones((num_constraints, 1))
    return c_mat, c_sol

def which_voxel(p, box_dim, res):
    if len(p.shape) == 1:
        p = p[np.newaxis, :]

    ix = np.maximum(np.minimum(np.floor(p[:,0]/box_dim), res-1), 0);
    iy = np.maximum(np.minimum(np.floor(p[:,1]/box_dim), res-1), 0);
    iz = np.maximum(np.minimum(np.floor(p[:,2]/box_dim), res-1), 0) if p.shape[1] == 3 else 0;

    ind = iz*(res**2) + iy*res + ix;
    return ind.astype(int)

def intersect_line(p0, p1, i0, i1, box_dim, res):
    if p0.size == 0:
        return None
    axis = np.abs(i0-i1) // res
    axis[axis > 1] = 2 if p0.shape[1] == 3 else 1

    i_max = np.maximum(i0, i1)
    idx = box_dim*np.array(np.unravel_index(i_max, (res,)*p0.shape[1], order='F'))
    t = ((idx.T - p0) / (p1 - p0))[np.arange(p0.shape[0]), axis]

    npos = p0 + (p1-p0)*t[:, np.newaxis]
    notvalid = np.any(~np.isfinite(npos), axis=1)
    if np.any(notvalid):
        print('p0', p0[notvalid][0])
        print('p1', p1[notvalid][0])
        print('i0', i0[notvalid][0])
        print('i1', i1[notvalid][0])
        print('axis', axis[notvalid][0])
        print(idx[:, notvalid][:, 0])
        print(t[notvalid][0])
        print(box_dim)
        raise ValueError()

    return npos


def deflection_solve_gradient(phi, deflection, damp=0):
    gradients = []
    for i in range(deflection.shape[1]):
        gradients.append(linalg.lsqr(phi, deflection[:, i], damp, show=False))
    return gradients

def gradient_integration(diff_mats, constraints, gradients, damp=0):
    full_A = constraints[0]
    full_b = constraints[1]
    for i in range(len(diff_mats)):
        full_A = vstack([full_A, diff_mats[i]])
        full_b = np.vstack([full_b, gradients[i][0][:, np.newaxis]])

    return linalg.lsqr(full_A, full_b, damp, show=False)

def deflection_solve(defl_mat, constraints, deflection, damp=0.):
    full_A = vstack([constraints[0], defl_mat])

    full_b = constraints[1]
    full_b = np.vstack([full_b, np.reshape(deflection, (-1, 1), order='F')])

    # TODO(ateh): Tryout conjugate gradient instead
    # A = full_A.transpose().dot(full_A)
    # b = full_A.transpose().dot(full_b)
    # return linalg.cg(A, b, x0=np.ones((full_A.shape[1],)), tol=1e-3, M=None, callback=None, atol=None)
    result = linalg.lsqr(full_A, full_b, damp, show=True)#, x0=1.0003*np.ones((full_A.shape[1],)))
    print('norm: {}'.format(result[3]/np.linalg.norm(full_b)))
    return result

def deflection_solve_lin_op(defl_mat, constraints, deflection, damp=0., x0=None):
    full_A = vstack([constraints[0], defl_mat]).tocsr()

    b = constraints[1]
    b = np.vstack([b, np.reshape(deflection, (-1, 1), order='F')])

    shape = full_A.shape
    A = LinearOperator((shape[1], shape[1]), lambda x : full_A.T.dot(full_A.dot(x)) - damp*x)
    result = linalg.cg(A, full_A.T.dot(b), tol=1e-10, x0=x0)
    res = full_A.dot(result[0]) - b.squeeze()
    res_act = np.linalg.norm(res)/np.linalg.norm(b.squeeze())
    # print('res actual: {}'.format(np.linalg.norm(full_A.T.dot(res))/np.linalg.norm(full_A.T.dot(b.squeeze()))))
    # print('res old: {}'.format(res_act))

    return result, res_act

def tof_solve(phi, tof, damp=0.):
    return linalg.lsqr(phi, tof, damp)

def get_linear_path(p0, d0, p1, d1, t):
    pos = (1-t)*p0 + t*p1
    return pos

def get_hermite_path(p0, d0, p1, d1, t):
  v = ( 2*t**3 - 3*t**2 + 1)*p0 + \
      (   t**3 - 2*t**2 + t)*d0 + \
      (-2*t**3 + 3*t**2    )*p1 + \
      (   t**3 -   t**2)*d1;
  return v

def get_true_path(p0, d0, p1, d1, t, path, path_start, path_end):
    num_rays = p0.shape[0]
    idx = t*(path_end-path_start) + path_start
    idx_l = np.floor(idx).astype(int)
    idx_h = np.ceil(idx).astype(int)
    a = idx_h - idx
    
    idx_l = num_rays*idx_l + np.arange(0, num_rays)
    idx_h = num_rays*idx_h + np.arange(0, num_rays)
   
    if np.any(a < 0) or np.any(a > 1):
        print("bad vals!")
    a = a[:, None]
    pos = a*path[idx_l, :] + (1-a)*path[idx_h, :]
    return pos