import torch
from grid import Grid


def generate_sensor(rays, e, plane, res, span, tangent=None):
    # consider only squares for now
    x, v = trace_rays_to_plane(rays, plane)
    p, n = plane

    t, t2 = get_tan_vecs(n, tangent)
    T = torch.t(torch.cat([t, t2], dim=0))
    h = span / res
    zeros = torch.zeros((res,)*(x.shape[1]-1), device=x.device, dtype=x.dtype)
    sensor = Grid(zeros, h)

    # do we add foreshortening?
    fs = torch.matmul(v[:, None, :], n[:, :, None]).squeeze(2).squeeze(1)
    fs = torch.abs(fs)

    # reject n from x
    xn = torch.matmul((x - p)[:, None, :], T[None, :, :]).squeeze(1)
    xn += (span/2)
    # need to add offset

    sensor.Splat(xn, fs*e, average=False)

    # we don't actually want to render with the weights, just let it pass through
    return sensor.scene


def generate_inf_sensor(rays, e, plane, res, angle_span=120, tangent=None):
    # we really only care about v
    x, v = rays
    p, n = plane

    v_norm = v / torch.norm(v, dim=-1, keepdim=True)

    ang_cut = torch.sin(0.5*torch.deg2rad(torch.tensor(angle_span, dtype=x.dtype)))

    t1, t2 = get_tan_vecs(n, tangent)
    T = torch.t(torch.cat([t1, t2], dim=0))

    zeros = torch.zeros((res,)*(x.shape[1]-1), device=x.device, dtype=x.dtype)
    sensor = Grid(zeros, 2*ang_cut/res)

    vn = torch.matmul(v_norm[:, None, :], T[None, :, :]).squeeze(1)
    vn += ang_cut

    fe = e*torch.ones(x.shape[0])

    sensor.Splat(vn, fe, average=False)

    return sensor.scene


def generate_pleno_sensor(rays, e, plane, bins, span, angle_span=120, tangent=None):
    x, v = trace_rays_to_plane(rays, plane)
    p, n = plane

    h = span / bins[0]
    ang_cut = torch.sin(0.5*torch.deg2rad(torch.tensor(angle_span, dtype=x.dtype)))

    t1, t2 = get_tan_vecs(n, tangent)
    Tx = torch.t(torch.cat([t1, t2], dim=0))
    Tv = torch.t(torch.cat([t1, -t2], dim=0))

    xgrid = Grid(torch.zeros(bins[0], bins[1]), h)
    vgrid = Grid(torch.zeros(bins[2], bins[3]), 2*ang_cut/bins[2])

    xn = torch.matmul((x - p)[:, None, :], Tx[None, :, :]).squeeze(1)
    xn += (span/2)

    vn = torch.matmul(v[:, None, :], Tv[None, :, :]).squeeze(1)
    vn += ang_cut

    _, rx, _, xidx = xgrid.index_values(xn)
    _, rv, _, vidx = vgrid.index_values(vn)

    del vn, xn

    ids = torch.stack(xidx)
    xmask = torch.all((ids >= 0) & (ids < bins[0]), dim=0)
    xib = [i[xmask] for i in xidx]
    vib = [torch.clamp(i[xmask], min=0, max=(bins[2]-1)) for i in vidx]
    iib = xib + vib

    wx, _, _ = Grid.rbf_tent(rx)
    wv, _, _ = Grid.rbf_tent(rv)

    wxe = (wx/wx.sum(dim=1, keepdim=True))[xmask]
    wve = (wv/wv.sum(dim=1, keepdim=True))[xmask]

    fs = torch.abs(torch.matmul(v[:, None, :], n[:, :, None]).squeeze(2).squeeze(1))
    fe = e*fs
    fe = (fe[:, None].expand_as(wx))[xmask]

    pleno = torch.zeros(*bins)
    pleno.index_put_(iib, wxe*wve*fe, accumulate=True)
    return pleno


def get_sdf_vals_near(rays, d_tex, plane, span, tangent=None):
    x, v = trace_rays_to_plane(rays, plane)
    p, n = plane

    res = d_tex.shape[0]
    h = span / res

    x_grid = Grid(d_tex, h)

    t, t2 = get_tan_vecs(n, tangent)
    T = torch.t(torch.cat([t, t2], dim=0))

    # reject n from x
    xn = torch.matmul((x - p)[:, None, :], T[None, :, :]).squeeze(1)
    xn += (span/2)

    disp_x, _ = x_grid.Get(xn)
    return disp_x


def get_sdf_vals_far(rays, d_tex, plane, ang_span, tangent=None):
    x, v = trace_rays_to_plane(rays, plane)
    p, n = plane

    res = d_tex.shape[0]

    ang_cut = torch.sin(0.5*torch.deg2rad(torch.tensor(ang_span, dtype=x.dtype)))
    h = 2*ang_cut/res

    t1, t2 = get_tan_vecs(n, tangent)
    T = torch.t(torch.cat([t1, t2], dim=0))

    vn = torch.matmul(v[:, None, :], T[None, :, :]).squeeze(1)
    vn += ang_cut

    x_grid = Grid(d_tex, h)
    defl_x, _ = x_grid.Get(vn)
    return defl_x


def get_disps_from_tex(rays, d_tex, plane, span, tangent=None):
    x, v = trace_rays_to_plane(rays, plane)
    p, n = plane

    res = d_tex.shape[0]
    h = span / res

    x_grid = Grid(d_tex[..., 0], h)
    y_grid = Grid(d_tex[..., 1], h)

    t, t2 = get_tan_vecs(n, tangent)
    T = torch.t(torch.cat([t, t2], dim=0))

    # reject n from x
    xn = torch.matmul((x - p)[:, None, :], T[None, :, :]).squeeze(1)
    xn += (span/2)

    disp_x, _ = x_grid.Get(xn)
    disp_y, _ = y_grid.Get(xn)

    disps = torch.stack([disp_x, disp_y], dim=-1) - (span/2)

    disps3 = torch.matmul(T[None, :, :], disps[:, :, None]).squeeze(2)
    return disps3 + p


def get_defls_from_tex(rays, d_tex, plane, span, tangent=None):
    x, v = trace_rays_to_plane(rays, plane)
    p, n = plane

    res = d_tex.shape[0]
    h = span / res

    x_grid = Grid(d_tex[..., 0], h)
    y_grid = Grid(d_tex[..., 1], h)

    t, t2 = get_tan_vecs(n, tangent)
    T = torch.t(torch.cat([t, t2], dim=0))

    # reject n from x
    xn = torch.matmul((x - p)[:, None, :], T[None, :, :]).squeeze(1)
    xn += span / 2

    defl_x = 2*(x_grid.Get(xn)[0] - 0.5)
    defl_y = 2*(y_grid.Get(xn)[0] - 0.5)
    defl_z = 1 - defl_x**2 - defl_y**2

    defls = torch.stack([defl_x, defl_y, defl_z], dim=-1)
    frame = torch.t(torch.cat([t, t2, n], dim=0))

    return torch.matmul(frame[None, :, :], defls[:, :, None]).squeeze(2)


def trace_rays_to_plane(rays, plane):
    x, v = rays
    p, n = plane

    t = torch.matmul(n[:, None, :], (p - x)[:, :, None]).squeeze(2)
    t /= torch.matmul(n[:, None, :], v[:, :, None]).squeeze(2)

    return (x + t*v), v


def refract(rays, plane, etai, etae=1.0):
    x, v = rays
    p, n = plane

    cosi = torch.matmul(v[:, None, :], n[:, :, None]).squeeze()
    eta = etai / etae

    k = 1 - eta**2 * (1 - cosi**2)

    vout = torch.zeros_like(v)
    vout[k < 0] = 0
    vout[k >= 1] = eta*v + (eta * cosi[:, None] - torch.sqrt(k)) * torch.sign(cosi) * n

    return x, vout


def get_tan_vecs(n, t=None):
    if t is None:
        t2 = torch.zeros_like(n)
        if torch.abs(n)[0, -1] > 0.001:
            t2[0, 0] = 1
        else:
            t2[0, -1] = 1
    else:
        t2 = t
    t1 = torch.cross(n, t2, dim=1)
    return t1, t2
