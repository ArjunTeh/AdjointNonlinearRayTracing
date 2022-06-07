import torch
import numpy as np
from functools import partial


def plane_source(angle, num_rays, width):
    x = torch.stack([torch.linspace(-width/2, width/2, num_rays),
                     torch.zeros(num_rays)], dim=1)
    v = torch.tensor([[0., 1.]]).repeat(x.shape[0], 1)

    v = rotate_ray(v, angle)
    x = rotate_ray(x, angle) + width/2
    x -= np.sqrt(2)*width*v/2

    plane_v = v.clone()
    plane_x = np.sqrt(2)*width*v/2 + width/2

    planes = torch.stack([plane_x, plane_v], dim=1)
    # rotate the rays
    return x, v, planes


def plane_source3(angle, num_rays, width, circle=False):
    pts = torch.meshgrid([torch.linspace(-width/2, width/2, num_rays)]*2)
    pts = [pts[0], torch.zeros(num_rays, num_rays), pts[1]]
    return rotate_pts_to_source(pts, angle, width, circle=circle)


def point_source3(angle, pixels, spp, width, cone_angle=90, xaxis=False, sensor_dist=0.0, circle=False):
    ang_rad = np.radians(cone_angle/2)
    spp = np.maximum(int(np.floor(np.sqrt(spp))), 1)
    theta, phi = torch.meshgrid([torch.linspace(-ang_rad, ang_rad, p*spp) for p in pixels])
    theta, phi = theta.flatten(), phi.flatten()
    vel = torch.stack([torch.cos(theta)*torch.sin(phi),
                       torch.cos(theta)*torch.cos(phi),
                       torch.sin(theta)], dim=-1)

    pos = torch.tensor([[0, -width/2, 0]]).repeat(theta.shape[0], 1)

    vel /= torch.norm(vel, dim=-1, keepdim=True)

    x = rotate_ray3(pos, angle, vert=xaxis) + width/2
    v = rotate_ray3(vel, angle, vert=xaxis)

    plane_v = torch.tensor([0., 1., 0.]).repeat(v.shape[0], 1)
    plane_v = rotate_ray3(plane_v, angle, vert=xaxis)
    plane_t = torch.tensor([0., 0., 1.]).repeat(v.shape[0], 1)
    plane_t = rotate_ray3(plane_t, angle, vert=xaxis)
    plane_x = (sensor_dist+width/2)*plane_v + width/2
    planes = torch.stack([plane_x, plane_v, plane_t], dim=1)
    return x, v, planes


def plane_source3_rand(angle, pixels, spp, width, circle=False, xaxis=False, sensor_dist=1.0, independent=False):

    offset = torch.rand(2*spp, pixels[0], pixels[1]) * width
    rng = [width*(torch.arange(p)/p - 0.5)
           for p in pixels]

    if independent:
        pts = [offset[:spp, ...] - (width/2),
               torch.zeros(spp, *pixels),
               offset[spp:, ...] - (width/2)]
    else:
        pts = torch.meshgrid(*rng)
        pts = [pts[0] + offset[:spp, ...] / pixels[0],
               torch.zeros(*pixels, spp),
               pts[1] + offset[spp:, ...] / pixels[1]]
    return rotate_pts_to_source(pts, angle, width, circle=circle, xaxis=xaxis, sensor_dist=sensor_dist)


def point_source3_rand(angle, pixels, spp, width, circle=False, xaxis=False, sensor_dist=1.0):
    offset = torch.rand(2*spp, pixels[0], pixels[1]) - 0.5

    rng = [width*((torch.arange(p)+0.5) / p - 0.5)
           for p in pixels]
    pts = torch.meshgrid(*rng)
    pts = [pts[0] + offset[:spp, ...],
           pts[1] + offset[spp:, ...]]

    if circle:
        mask = torch.norm(torch.stack([p.flatten() for p in pts]), dim=0)
        mask = mask < (width/2)

    vels = [pts[0],
            width*torch.ones(*pixels, spp),
            pts[1]]
    vel = torch.stack([p.flatten() for p in vels], dim=-1)
    vel = vel / torch.norm(vel, dim=-1, keepdim=True)

    if circle:
        vel = vel[mask]

    pos = torch.tensor([0.0, -width/2, 0.0]).repeat(vel.shape[0], 1)
    x = rotate_ray3(pos, angle, vert=xaxis) + width/2
    v = rotate_ray3(vel, angle, vert=xaxis)

    plane_v = torch.tensor([0., 1., 0.]).repeat(v.shape[0], 1)
    plane_v = rotate_ray3(plane_v, angle, vert=xaxis)
    plane_t = torch.tensor([0., 0., 1.]).repeat(v.shape[0], 1)
    plane_t = rotate_ray3(plane_t, angle, vert=xaxis)
    plane_x = sensor_dist*width*plane_v/2 + width/2
    planes = torch.stack([plane_x, plane_v, plane_t], dim=1)
    return x, v, planes


def area_source3_rand_bias(angle, pixels, spp, width, circle=False, xaxis=False, sensor_dist=1.0):
    offset = torch.rand(2*spp, pixels[0], pixels[1]) - 0.5
    offset *= width / pixels[0]
    rng = [width*((torch.arange(p)+0.5) / p - 0.5)
           for p in pixels]
    pts = torch.meshgrid(*rng)
    pts = [pts[0] + offset[:spp, ...],
           torch.zeros(*pixels, spp),
           pts[1] + offset[spp:, ...]]

    pos = torch.stack([p.flatten() for p in pts], dim=-1)
    if circle:
        mask = torch.norm(pos, dim=-1) < (width/2)
        pos = pos[mask]

    pt = -pos
    pos -= (sensor_dist + width/2) * torch.tensor([[0, 1, 0]])
    pt += (sensor_dist + width/2) * torch.tensor([[0, 1, 0]])

    tosense = torch.rand(2, pos.shape[0]) - 0.5
    tosense *= 1.0*width

    target = torch.stack([tosense[0, ...],
                          width*torch.ones(pos.shape[0])/2,
                          tosense[1, ...]], dim=-1)

    vel = target - pos
    vel /= torch.norm(vel, dim=-1, keepdim=True)

    tpv = sensor_dist / vel[..., 1]
    npos = pos + tpv[:, None]*vel

    xt = rotate_ray3(pt, angle, vert=xaxis) + width/2
    x = rotate_ray3(npos, angle, vert=xaxis) + width/2
    v = rotate_ray3(vel, angle, vert=xaxis)

    plane_v = torch.tensor([0., 1., 0.]).repeat(v.shape[0], 1)
    plane_v = rotate_ray3(plane_v, angle, vert=xaxis)
    plane_t = torch.tensor([0., 0., 1.]).repeat(v.shape[0], 1)
    plane_t = rotate_ray3(plane_t, angle, vert=xaxis)
    plane_x = (sensor_dist+width/2)*plane_v + width/2
    planes = torch.stack([plane_x, plane_v, plane_t], dim=1)

    return (x, v, planes), xt, tpv


def area_source3_cone(angle, pixels, spp, width, circle=False, xaxis=False, sensor_dist=1.0, cone_angle=90):
    offset = torch.rand(2*spp, pixels[0], pixels[1]) - 0.5
    offset *= width / pixels[0]
    rng = [width*((torch.arange(p)+0.5) / p - 0.5)
           for p in pixels]
    pts = torch.meshgrid(*rng)
    pts = [pts[0] + offset[:spp, ...],
           -width*torch.ones(*pixels, spp)/2,
           pts[1] + offset[spp:, ...]]

    pos = torch.stack([p.flatten() for p in pts], dim=-1)
    if circle:
        mask = torch.norm(pos, dim=-1) < (width/2)
        pos = pos[mask]

    forward = torch.zeros_like(pos)
    forward[:, 1] = 1
    vel = hatbox_sample(forward, cone_angle)
    tpv = sensor_dist / vel[..., 1]

    x = rotate_ray3(pos, angle, vert=xaxis) + width/2
    v = rotate_ray3(vel, angle, vert=xaxis)

    plane_v = torch.tensor([0., 1., 0.]).repeat(v.shape[0], 1)
    plane_v = rotate_ray3(plane_v, angle, vert=xaxis)
    plane_t = torch.tensor([0., 0., 1.]).repeat(v.shape[0], 1)
    plane_t = rotate_ray3(plane_t, angle, vert=xaxis)
    plane_x = (sensor_dist+width/2)*plane_v + width/2
    planes = torch.stack([plane_x, plane_v, plane_t], dim=1)

    return (x, v, planes), tpv


def cone_source3_rand(angle, pixels, spp, width, circle=False, xaxis=False, sensor_dist=1.0, cone_angle=100.0):
    pos = torch.tensor([[0, -width/2, 0]]).repeat(pixels[0]*pixels[1]*spp, 1)
    vel = torch.zeros_like(pos)
    vel[:, 1] = 1
    vel = hatbox_sample(vel, cone_angle)

    x = rotate_ray3(pos, angle, vert=xaxis) + width/2
    v = rotate_ray3(vel, angle, vert=xaxis)

    plane_v = torch.tensor([0., 1., 0.]).repeat(v.shape[0], 1)
    plane_v = rotate_ray3(plane_v, angle, vert=xaxis)
    plane_t = torch.tensor([0., 0., 1.]).repeat(v.shape[0], 1)
    plane_t = rotate_ray3(plane_t, angle, vert=xaxis)
    plane_x = (sensor_dist+width/2)*plane_v + width/2
    planes = torch.stack([plane_x, plane_v, plane_t], dim=1)

    return x, v, planes


def area_source3_rand(angle, pixels, spp, width, circle=False, xaxis=False, sensor_dist=1.0):
    posf = []
    velf = []
    ptf = []
    tpf = []
    nrays = 0

    for iteration in range(1):
        offset = torch.rand(2*spp, pixels[0], pixels[1]) - 0.5
        offset *= width / pixels[0]
        hemi = torch.normal(0, 1, size=(spp*pixels[0]*pixels[1], 3))

        rng = [width*((torch.arange(p)+0.5) / p - 0.5)
               for p in pixels]
        pts = torch.meshgrid(*rng)
        pts = [pts[0] + offset[:spp, ...],
               torch.zeros(*pixels, spp),
               pts[1] + offset[spp:, ...]]

        pos = torch.stack([p.flatten() for p in pts], dim=-1)

        vel = hemi / torch.norm(hemi, dim=-1, keepdim=True)
        vel[..., 1] = torch.abs(vel[..., 1])

        if circle:
            mask = torch.norm(pos, dim=-1) < (width/2)
            pos = pos[mask]
            vel = vel[mask]

        pt = -pos
        pos -= (sensor_dist + width/2) * torch.tensor([[0, 1, 0]])
        pt += (sensor_dist + width/2) * torch.tensor([[0, 1, 0]])

        # check to see if ray just misses the volume
        # tp = -pos[..., 1] / vel[..., 1]
        tpv = sensor_dist / vel[..., 1]
        # npos = pos + tp[:, None]*vel
        npos = pos + tpv[:, None]*vel
        hitvol = (torch.abs(npos) <= (width/2)).all(dim=-1)

        if not hitvol.any():
            raise ValueError('no rays')

        nrays += npos[hitvol].shape[0]
        posf.append(pos[hitvol])
        velf.append(vel[hitvol])
        ptf.append(pt[hitvol])
        tpf.append(tpv[hitvol])

        if nrays >= 0.55*spp*pixels[0]*pixels[1]:
            break
        
    ptf = torch.cat(ptf)
    posf = torch.cat(posf)
    velf = torch.cat(velf)
    tpf = torch.cat(tpf)
    xt = rotate_ray3(ptf, angle, vert=xaxis) + width/2
    x = rotate_ray3(posf, angle, vert=xaxis) + width/2
    v = rotate_ray3(velf, angle, vert=xaxis)

    plane_v = torch.tensor([0., 1., 0.]).repeat(v.shape[0], 1)
    plane_v = rotate_ray3(plane_v, angle, vert=xaxis)
    plane_t = torch.tensor([0., 0., 1.]).repeat(v.shape[0], 1)
    plane_t = rotate_ray3(plane_t, angle, vert=xaxis)
    plane_x = (sensor_dist+width/2)*plane_v + width/2
    planes = torch.stack([plane_x, plane_v, plane_t], dim=1)

    return (x, v, planes), xt, tpf


def rotate_pts_to_source(pts, angle, width, circle=False, xaxis=False, sensor_dist=1.0):
    x = torch.stack([p.flatten() for p in pts], dim=-1)
    if circle:
        r = torch.norm(x, dim=-1)
        # x = x[r < 0.5*width/2]
        x = x[r < width/2]
    v = torch.tensor([0.0, 1.0, 0.0]).repeat(x.shape[0], 1)
    t = torch.tensor([0.0, 0.0, 1.0]).repeat(x.shape[0], 1)

    x = rotate_ray3(x, angle, vert=xaxis) + width/2
    v = rotate_ray3(v, angle, vert=xaxis)
    t = rotate_ray3(t, angle, vert=xaxis)
    x -= (width)*v/2

    plane_v = v.clone()
    plane_x = (sensor_dist+(width/2))*v + width/2
    plane_t = t.clone()
    planes = torch.stack([plane_x, plane_v, plane_t], dim=1)
    return x, v, planes


def rotate_ray(x, angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = torch.tensor(((c, -s), (s, c)), device=x.device, dtype=x.dtype)
    return torch.matmul(x, R.T)


def rotate_ray3(x, angle, vert=False):
    theta = np.radians(angle.cpu())
    c, s = np.cos(theta), np.sin(theta)
    if vert:
        Rn = np.array(((1, 0, 0), (0, c, -s), (0, s, c))).astype(float)
    else:
        Rn = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))).astype(float)
    R = torch.from_numpy(Rn)
    R = R.to(device=x.device, dtype=x.dtype)
    return torch.matmul(x, R.T)


def sample_sphere(nrays, width, cone_angle=90.0, lens_type='luneburg'):
    x = torch.randn(nrays, 3)
    xn = x / torch.norm(x, dim=1, keepdim=True)
    v = -xn
    vn = hatbox_sample(v, cone_angle)
    xn = xn*width/2

    tangent = torch.randn(nrays, 3)
    # tangent_proj = torch.matmul(tangent[:, None, :], v[:, :, None]).squeeze()
    plane_t = tangent / torch.norm(tangent, dim=1, keepdim=True)

    if lens_type == 'luneburg':
        plane_x = (width/2) + vn*(width/2)
    else:
        plane_x = -xn + width/2
    plane_v = v
    planes = torch.stack([plane_x, plane_v, plane_t], dim=1)

    rpv = [nrays]
    return (xn + width/2, vn, planes), rpv


def rays_in_circle(nviews, rays_per_view, width, angle_span=360):
    angles = torch.linspace(0, angle_span, nviews + 1)
    view_list = [plane_source(angles[i], rays_per_view, width)
                 for i in range(nviews)]

    return tuple(map(torch.cat, zip(*view_list)))


def rays_in_sphere(nviews, rays_per_view, width, angle_span=360, circle=False):
    angles = torch.linspace(0, angle_span, nviews+1)
    view_list = [plane_source3(angles[i], rays_per_view, width, circle=circle)
                 for i in range(nviews)]
    return tuple(map(torch.cat, zip(*view_list)))


def rand_rays_in_sphere(nviews, im_res, spp, width, angle_span=360, circle=False, xaxis=False, sensor_dist=1.0, indep=False):
    angles = torch.linspace(0, angle_span, nviews+1)
    view_list = [plane_source3_rand(angles[i], im_res, spp, width, circle=circle, xaxis=xaxis, sensor_dist=sensor_dist, independent=indep)
                 for i in range(nviews)]
    nrays = [v[0].shape[0] for v in view_list]
    return tuple(map(torch.cat, zip(*view_list))), nrays


def rand_ptrays_in_sphere(nviews, im_res, spp, width, angle_span=360, circle=False, xaxis=False, sensor_dist=0.0):
    angles = torch.linspace(0, angle_span, nviews+1)
    view_list = [point_source3_rand(angles[i], im_res, spp, width, circle=circle, xaxis=xaxis, sensor_dist=sensor_dist)
                 for i in range(nviews)]
    nrays = [v[0].shape[0] for v in view_list]
    return tuple(map(torch.cat, zip(*view_list))), nrays


def rand_area_in_sphere(nviews, im_res, spp, width, angle_span=360, circle=False, xaxis=False, sensor_dist=1.0):
    angles = torch.linspace(0, angle_span, nviews+1)
    view_list = [area_source3_rand_bias(angles[i], im_res, spp, width, circle=circle, xaxis=xaxis, sensor_dist=sensor_dist)
                 for i in range(nviews)]
    # view_list = [area_source3_rand(angles[i], im_res, spp, width, circle=circle, xaxis=xaxis, sensor_dist=sensor_dist)
    #              for i in range(nviews)]
    views, targets, dists = zip(*view_list)
    nrays = [v[0].shape[0] for v in views]
    return tuple(map(torch.cat, zip(*views))), torch.cat(targets), torch.cat(dists), nrays


def rand_cone_in_sphere(nviews, im_res, spp, width, angle_span=360, circle=False, xaxis=False, sensor_dist=1.0, cone_angle=90.0):
    angles = torch.linspace(0, angle_span, nviews+1)
    view_list = [area_source3_cone(angles[i], im_res, spp, width, circle=circle, xaxis=xaxis, sensor_dist=sensor_dist, cone_angle=cone_angle)
                 for i in range(nviews)]
    views, dists = zip(*view_list)
    nrays = [v[0].shape[0] for v in views]
    return tuple(map(torch.cat, zip(*views))), torch.cat(dists), nrays


def rand_ptcone_in_sphere(nviews, im_res, spp, width, angle_span=360, circle=False, xaxis=False, sensor_dist=1.0, cone_angle=90.0):
    angles = torch.linspace(0, angle_span, nviews+1)
    view_list = [cone_source3_rand(angles[i], im_res, spp, width, circle=circle, xaxis=xaxis, sensor_dist=sensor_dist, cone_angle=cone_angle)
                 for i in range(nviews)]
    views = list(zip(*view_list))
    nrays = [v[0].shape[0] for v in views]
    dists = torch.zeros(nviews)
    return tuple(map(torch.cat, views)), dists, nrays


def rand_rays_cube(im_res, spp, width, circle=False, src_type='plane', cone_ang=90):
    if src_type == 'plane':
        src_gen = plane_source3_rand
    elif src_type == 'point':
        src_gen = partial(point_source3, cone_angle=cone_ang)
    else:
        src_gen = partial(cone_source3_rand, cone_angle=cone_ang)
    angles = torch.linspace(0, 360, 5)
    vangles = torch.tensor([90, -90])
    view_list = [src_gen(angles[i], im_res, spp, width, circle=circle, xaxis=False, sensor_dist=0.0)
                 for i in range(len(angles)-1)]
    view_list.extend([src_gen(va, im_res, spp, width, circle=circle, xaxis=True, sensor_dist=0.0)
                      for va in vangles])
    nrays = [v[0].shape[0] for v in view_list]
    return tuple(map(torch.cat, zip(*view_list))), nrays


def sum_norm(im, scale=False):
    npix = torch.numel(im)
    scalar = npix / im.sum()
    if scale:
        return scalar*im, scalar
    return scalar*im


def sum_norm2(im, scale=False):
    npix = torch.numel(im)
    scalar = npix / torch.norm(im)
    if scale:
        return scalar*im, scalar
    return scalar*im


def norm_image(im):
    rng = im.max() - im.min()
    if torch.isclose(rng, torch.zeros_like(rng)):
        return im

    return (im - im.min()) / (im.max() - im.min())


def tent_filter(x, r=1):
    inv_dist = r - x
    dx = -torch.ones_like(x)
    dx[inv_dist < 0] = 0
    return inv_dist.clamp(min=0), dx


def gauss_filter(x, r=1.0, a=0.5):
    if torch.all(x >= 1):
        print(x)
        raise ValueError("stop")
    v = torch.exp(-a*(x**2)) - np.exp(-a*(r**2))
    vx = -2*a*x*torch.exp(-a*(x**2)) - np.exp(-a*(r**2))
    mask = torch.abs(x) > 1
    v[mask] = 0
    vx[mask] = 0
    return v, vx


def create_sensor(x, v, plane, nbins, span, e=1):
    p, n, t = plane[None, 0], plane[None, 1], rotate_ray(plane[None, 1], 90)
    h = span / nbins

    dp = torch.matmul((x - p)[:, None, :], t[:, :, None])
    dpn = nbins * (0.5 + (dp / span)) - 0.5
    dpn = dpn.squeeze(2).squeeze(1)

    at = torch.matmul(v[:, None, :], n[:, :, None]).squeeze(2).squeeze(1)
    at = torch.ones_like(dpn)
    vals = torch.abs(e*at)

    dpl = torch.floor(dpn).long()
    dph = dpl + 1

    lm = (dpl < nbins) & (dpl >= 0)
    hm = (dph < nbins) & (dph >= 0)

    # wl, wlx = gauss_filter(dpn[lm] - dpl[lm])
    # wh, whx = gauss_filter(dpn[hm] - dph[hm])
    wl, wlx = tent_filter(dpn - dpl)
    wh, whx = tent_filter(dpn - dph)
    ws = wl + wh

    sensor = torch.zeros(nbins, device=x.device, dtype=vals.dtype)
    sensor.index_put_((dpl[lm],), (wl*vals/ws)[lm], accumulate=True)
    sensor.index_put_((dph[hm],), (wh*vals/ws)[hm], accumulate=True)

    # ws = torch.zeros(nbins, device=x.device, dtype=vals.dtype)
    # ws.index_put_((dpl[lm],), wl, accumulate=True)
    # ws.index_put_((dph[hm],), wh, accumulate=True)
    # ws[ws < e*1e-6] = 1
    # sensor /= ws
    # sensor /= sensor.sum()

    sv = torch.zeros_like(v)
    sv[lm] += wl[lm, None]*n
    sv[hm] += wh[hm, None]*n

    sx = torch.zeros_like(x)
    sx[lm] += (wlx*vals)[lm, None]*t / h
    sx[hm] += (whx*vals)[hm, None]*t / h

    return sensor, (sx, sv, dpl.clamp(0, nbins-1), dph.clamp(0, nbins-1))


def render_intensities(x, v, planes, nviews, nrays, nbins, dim, grad=False):
    # nrays is stride, nviews is number of planes to generate
    xp = x.split(nrays)
    vp = v.split(nrays)
    p = planes[::nrays]

    out = [create_sensor(xp[i], vp[i], p[i], nbins, dim, e=(1/nrays))
           for i in range(nviews)]
    out = list(zip(*out))

    ims = torch.cat(out[0])
    dxs = list(map(torch.cat, zip(*out[1])))
    if grad:
        return ims, dxs
    return ims


def perturb_vector(v, spp):
    P = torch.randn(v.shape[0]*spp, v.shape[1])
    P /= P.norm(dim=-1, keepdim=True)

    vn = v.repeat(spp, 1) + P
    vn /= vn.norm(dim=-1, keepdim=True)

    return vn


def hatbox_sample(v, angle):
    basis = torch.tensor([[0, 0, 1.0]])
    rang = torch.deg2rad(torch.tensor(angle)) / 2
    dist = torch.cos(rang)
    z = torch.rand(v.shape[0])*(1-dist) + dist
    theta = 2*np.pi*torch.rand(v.shape[0])
    scale = torch.sqrt(1-(z**2))

    x = torch.cos(theta) * scale
    y = torch.sin(theta) * scale

    t1 = torch.cross(basis.expand_as(v), v, dim=-1)
    t2 = torch.cross(t1, v, dim=-1)

    return x[:, None] * t1 + y[:, None] * t2 + z[:, None] * v


def random_rotmat():
    from scipy.spatial.transform import Rotation as R
    rot_mat = R.random().as_matrix()
    trot = torch.from_numpy(rot_mat)
    return trot


def random_rotate_ic(x, v, planes, span):
    rotmat = random_rotmat().to(device=x.device, dtype=x.dtype)
    xn = torch.matmul(rotmat, x[..., None] - (span/2)) + (span/2)
    vn = torch.matmul(rotmat, v[..., None])
    sp = torch.matmul(rotmat, planes[:, 0, :, None] - (span/2)) + (span/2)
    sn = torch.matmul(rotmat, planes[:, 1, :, None])
    st = torch.matmul(rotmat, planes[:, 2, :, None])

    return xn.squeeze(-1), vn.squeeze(-1), torch.stack([sp.squeeze(-1), sn.squeeze(-1), st.squeeze(-1)], dim=1)


def rotate_ic(x, v, planes, angle, span, vert=False):
    xr = rotate_ray3(x, angle, vert=vert) + (span/2)
    vr = rotate_ray3(v, angle, vert=vert)
    spr = rotate_ray3(planes[:, 0, :], angle, vert=vert) + (span/2.0)
    snr = rotate_ray3(planes[:, 1, :], angle, vert=vert)
    str = rotate_ray3(planes[:, 2, :], angle, vert=vert)

    return xr, vr, torch.stack([spr, snr, str], dim=1)
