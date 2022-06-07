import torch


class Grid:

    def __init__(self, scene, h, hinv=None):
        self.scene = scene
        self.weights = torch.zeros_like(scene)
        self.res = scene.size()
        self.h = h
        self.hinv = hinv
        self.device = scene.device

    def check_input(self, x):
        if x.device != self.device:
            raise ValueError("input on device: {}, grid on device: {}"
                             .format(x.device, self.device))

        if x.shape[1] != self.scene.ndim:
            raise ValueError("input({}) is not the same dimension as grid({})"
                             .format(x.shape[1], self.scene.ndim))

    def bounds(self, x):
        neg = torch.all(x >= 0, dim=1)
        pos = x[:, 0] <= self.res[0]
        pos = pos & (x[:, 1] <= self.res[1])

        return (neg & pos)

    def render(self):
        n = self.scene.detach().clone()
        mask = ~torch.isclose(self.weights, torch.zeros_like(n))
        n[mask] /= self.weights[mask]
        return n

    # RBF implementation
    def index_values(self, x):
        norm_x = (x / self.h) - 0.5

        x1 = torch.floor(norm_x).long()
        x0 = x1 - 1
        x2 = x1 + 1
        x3 = x1 + 2

        idx = torch.stack([x0, x1, x2, x3]).split(1, dim=-1)

        mesh_idx = [torch.flatten(x)
                    for x in torch.meshgrid((torch.arange(4),)*x.shape[1])]

        indices = [ind.squeeze(-1)[x] for ind, x in zip(idx, mesh_idx)]
        capped = [torch.clip(x, 0, self.res[0]-1) for x in indices]
        fi = torch.t(self.scene[capped])
        dx = norm_x - torch.stack(indices, dim=-1)
        dx = dx.permute(1, 0, 2)

        r = torch.linalg.norm(dx, dim=-1)
        r0 = r.where(~torch.isclose(r, torch.zeros_like(r)),
                     torch.tensor(1, dtype=r.dtype, device=r.device))
        # r0[torch.isclose(r, torch.zeros_like(r))] = 1
        dx_nm = dx / r0[:, :, None]

        fin = fi

        return fin, r, dx_nm, list(map(torch.t, indices))

    def RenderGradient(self, linear=False):
        dev = self.scene.device
        idx = torch.meshgrid(*[self.h*torch.arange(r, device=dev) for r in self.res])
        z = torch.stack([x.flatten() for x in idx], dim=-1)
        if linear:
            f, fx = self.GetLinear(z)
        else:
            f, fx = self.Get(z)
        return fx.reshape(*self.res, self.scene.ndim)

    @staticmethod
    def rbf_tent(r):
        rt2 = (2*torch.ones(1, dtype=r.dtype, device=r.device)).sqrt()
        w = torch.clamp(rt2 - r, min=0)
        wx = -(r < rt2).to(r.dtype)
        return w, wx, 0

    @staticmethod
    def rbf_cubic(r):
        s = torch.sign(r)
        r = torch.abs(r)
        vals = torch.zeros_like(r)
        vx = torch.zeros_like(r)

        m12 = (r > 1) & (r < 2)
        vals[m12] = (1/6)*(2-r[m12])**3
        vx[m12] = -s[m12]*0.5*(2 - r[m12])**2

        m1 = r <= 1
        vals[m1] = (2/3) - r[m1]**2 + 0.5*r[m1]**3
        vx[m1] = s[m1]*(-2*r[m1] + (1.5)*r[m1]**2)

        return vals, vx, 0

    def Get(self, x, sigmoid=False, cubic=False):
        self.check_input(x)
        # prune x to correct values
        fi, r, dx, _ = self.index_values(x)

        if cubic:
            w, wx, wxx = Grid.rbf_cubic(r)
        else:
            w, wx, wxx = Grid.rbf_tent(r)

        ws = w.sum(dim=1)

        f = torch.matmul(fi[:, None, :], w[:, :, None]).squeeze(2)
        f /= ws[:, None]

        fx = ((wx*fi)[:, :, None]*dx).sum(dim=1)
        fx -= f * (wx[:, :, None]*dx).sum(dim=1)
        fx /= ws[:, None]

        if sigmoid:
            sf = torch.sigmoid(f.squeeze(1))
            sfx = (sf[:, None]**2) * torch.exp(-f) * fx / self.h
            return sf + 1, sfx

        return f.squeeze(1), fx / self.h

    def GetHessian(self, x):
        from torch.autograd.functional import jacobian as jac

        def myf(p):
            return self.Get(p)
        return jac(myf, x)

    def Splat(self, x, f, average=True):
        self.check_input(x)

        fi, r, dx, idx = self.index_values(x)
        w, wx, _ = Grid.rbf_tent(r)

        ids = torch.stack(idx)
        mask = torch.all((ids >= 0) & (ids < self.res[0]), dim=0)
        iib = [i[mask] for i in idx]

        fe = (f[:, None].expand_as(w))[mask]

        if not average:
            we = (w/w.sum(dim=1, keepdim=True))[mask]
        else:
            we = w[mask]

        self.scene.index_put_(iib, we*fe, accumulate=True)
        self.weights.index_put_(iib, we, accumulate=True)

    def SplatGrad(self, x, f, fx):
        self.check_input(x)

        r = torch.norm(fx, dim=-1)
        r0 = r.where(~torch.isclose(r, torch.zeros_like(r)),
                     torch.tensor(1, dtype=r.dtype, device=r.device))
        dx = self.h*(fx / r0[:, None])
        ff = self.h*(f + r)
        fb = self.h*(f - r)
        self.Splat(x, f)
        self.Splat(x+dx, ff)
        self.Splat(x-dx, fb)

    def SolveGrad(self, x, f, fx):
        self.check_input(x)
        fi, r, dx, idx = self.index_values(x)
        w, wx, wxx = Grid.rbf_tent(r)
        ws = w.sum(dim=1)

        a1 = wx[:, :, None] * dx
        a2 = w[:, :, None] * (torch.matmul(wx[:, None, :], dx) / (ws[:, None, None]))
        M = torch.cat([w[:, :, None], a1-a2], dim=-1).permute(0, 2, 1) / ws[:, None, None]
        b = torch.cat([f[:, None], fx], dim=-1)

        Mi = torch.pinverse(M)
        v = torch.matmul(Mi, b[:, :, None]).squeeze(2)

        mask = torch.stack(idx)
        mask = torch.all((mask >= 0) & (mask < self.res[0]), dim=0)
        iib = [x[mask] for x in idx]

        self.scene.index_put_(iib, v[mask], accumulate=True)
        self.weights.index_put_(iib, torch.ones_like(v[mask]), accumulate=True)

    def GetSpline(self, x):
        self.check_input(x)
        norm_x = (x / self.h)

        x0 = torch.floor(norm_x).long()

        w0 = Grid.rbf_cubic(norm_x - x0 + 1)
        w1 = Grid.rbf_cubic(norm_x - x0)
        w2 = Grid.rbf_cubic(norm_x - x0 - 1)
        w3 = Grid.rbf_cubic(norm_x - x0 - 2)

        idx = torch.stack([x0-1, x0, x0+1, x0+2]).split(1, dim=-1)
        weights = torch.stack([w0[0], w1[0], w2[0], w3[0]]).split(1, dim=-1)
        weights_dx = torch.stack([w0[1], w1[1], w2[1], w3[1]]).split(1, dim=-1)
        mesh_idx = [torch.flatten(x)
                    for x in torch.meshgrid((torch.arange(4),)*x.shape[1])]

        indices = [ind.squeeze(-1)[x] for ind, x in zip(idx, mesh_idx)]
        w_ind = [torch.clip(w.squeeze(-1)[x], 0, 1) for w, x in zip(weights, mesh_idx)]
        w_indx = [w.squeeze(-1)[x] for w, x in zip(weights_dx, mesh_idx)]
        capped = [torch.clip(x, 0, self.res[0]-1) for x in indices]
        fi = torch.t(self.scene[capped])
        wi = torch.t(torch.stack(w_ind, dim=-1).prod(dim=-1))

        f = torch.matmul(fi[:, None, :], wi[:, :, None]).squeeze(2).squeeze(1)

        wdx = torch.t(w_ind[1]*w_ind[2]*w_indx[0])
        wdy = torch.t(w_ind[2]*w_ind[0]*w_indx[1])
        wdz = torch.t(w_ind[0]*w_ind[1]*w_indx[2])
        wx = [wdx, wdy, wdz]

        # wx = [torch.where(mx == 0, -1, 1)[None, :]*torch.t(w)
        #       for w, mx in zip(reversed(w_ind), mesh_idx)]

        fx = torch.stack([torch.matmul(fi[:, None, :], w[:, :, None]).squeeze(2).squeeze(1)
                          for w in wx], dim=-1)
        return f, fx / self.h


    # Bi/Trilinear interpolation implementation
    def GetLinear(self, x, debug_print=False):
        self.check_input(x)
        if self.hinv is not None:
            norm_x = x*self.hinv
        else:
            norm_x = (x / self.h)

        x0 = torch.floor(norm_x).long()
        w0 = torch.clip(norm_x - x0, 0, 1)

        idx = torch.stack([x0, x0+1]).split(1, dim=-1)
        weights = torch.stack([1-w0, w0]).split(1, dim=-1)
        mesh_idx = [torch.flatten(x)
                    for x in torch.meshgrid((torch.arange(2),)*x.shape[1])]

        indices = [ind.squeeze(-1)[x] for ind, x in zip(idx, mesh_idx)]
        w_ind = [torch.clip(w.squeeze(-1)[x], 0, 1) for w, x in zip(weights, mesh_idx)]
        capped = [torch.clip(x, 0, self.res[0]-1) for x in indices]
        fi = torch.t(self.scene[capped])
        wi = torch.t(torch.stack(w_ind, dim=-1).prod(dim=-1))

        f = torch.matmul(fi[:, None, :], wi[:, :, None]).squeeze(2).squeeze(1)

        # TODO(ateh): make dimension agnostic
        if x.shape[1] == 2:
            wdx = torch.t(w_ind[1])*torch.where(mesh_idx[0]==0, -1, 1).to(device=x.device)
            wdy = torch.t(w_ind[0])*torch.where(mesh_idx[1]==0, -1, 1).to(device=x.device)
            wx = [wdx, wdy]
        elif x.shape[1] == 3:
            wdx = torch.t(w_ind[1]*w_ind[2])*(torch.where(mesh_idx[0]==0, -1, 1).to(device=x.device))
            wdy = torch.t(w_ind[2]*w_ind[0])*(torch.where(mesh_idx[1]==0, -1, 1).to(device=x.device))
            wdz = torch.t(w_ind[0]*w_ind[1])*(torch.where(mesh_idx[2]==0, -1, 1).to(device=x.device))
            wx = [wdx, wdy, wdz]

        # wx = [torch.where(mx == 0, -1, 1)[None, :]*torch.t(w)
        #       for w, mx in zip(reversed(w_ind), mesh_idx)]

        if debug_print:
            print('weights')
            print(norm_x)
            for i in range(fi.numel()):
                print(indices[0][i, 0], indices[1][i, 0], indices[2][i, 0], ":", fi[0, i])
                print(w_ind[0][i, 0], w_ind[1][i, 0], w_ind[2][i, 0])

        fx = torch.stack([torch.matmul(fi[:, None, :], w[:, :, None]).squeeze(2).squeeze(1)
                          for w in wx], dim=-1)
        return f, fx / self.h

    def SplatLinear(self, x, f, fx):
        self.check_input(x)
        norm_x = (x / self.h)

        x0 = torch.floor(norm_x).long()
        w0 = torch.clip(norm_x - x0, 0, 1)

        idx = torch.stack([x0, x0+1]).split(1, dim=-1)
        weights = torch.stack([1-w0, w0]).split(1, dim=-1)
        mesh_idx = [torch.flatten(x)
                    for x in torch.meshgrid((torch.arange(2),)*x.shape[1])]

        indices = [ind.squeeze(-1)[x] for ind, x in zip(idx, mesh_idx)]
        w_ind = [torch.clip(w.squeeze(-1)[x], 0, 1) for w, x in zip(weights, mesh_idx)]
        wp = torch.stack(w_ind).prod(dim=0)

        if x.shape[1] == 2:
            wdx = w_ind[1]*torch.where(mesh_idx[0] == 0, -1, 1).to(device=x.device)[:, None]
            wdy = w_ind[0]*torch.where(mesh_idx[1] == 0, -1, 1).to(device=x.device)[:, None]
            wi = torch.stack([wdx, wdy], dim=-1)
        elif x.shape[1] == 3:
            wdx = w_ind[1]*w_ind[2]*(torch.where(mesh_idx[0] == 0, -1, 1).to(device=x.device))[:, None]
            wdy = w_ind[2]*w_ind[0]*(torch.where(mesh_idx[1] == 0, -1, 1).to(device=x.device))[:, None]
            wdz = w_ind[0]*w_ind[1]*(torch.where(mesh_idx[2] == 0, -1, 1).to(device=x.device))[:, None]
            wi = torch.stack([wdx, wdy, wdz], dim=-1)
        else:
            raise NotImplementedError("n-linear interpolation only supports 2 and 3 dimensions")
        # wi = torch.stack(w_ind, dim=-1)

        mask = torch.all((norm_x >= 0) & (norm_x < self.res[0]), dim=-1)

        fe = f[mask].expand(2**x.shape[1], -1)
        fxe = fx[mask].expand(2**x.shape[1], -1, -1)
        dot = self.h*torch.matmul(fxe[:, :, None, :], wi[:, mask, :, None]).squeeze(3).squeeze(2)

        iib = [torch.clip(ix[:, mask], 0, self.res[0]-1) for ix in indices]

        # self.scene.index_put_(iib, fe + dot, accumulate=True)
        # self.weights.index_put_(iib, torch.ones_like(fe), accumulate=True)
        self.scene.index_put_(iib, (wp[:, mask]*fe) + dot, accumulate=True)
        self.weights.index_put_(iib, wp[:, mask], accumulate=True)


def upres_volume(n, new_res):
    nvox = torch.clip(torch.tensor(n.shape[0]-1), min=1)
    gt = Grid(n, 1 / nvox)
    idx = [torch.linspace(0, 1, s, device=n.device, dtype=n.dtype)
           for s in new_res]
    xyz = torch.meshgrid(*idx)
    x = torch.stack([ix.flatten() for ix in xyz], dim=-1)
    vals = gt.GetLinear(x)
    s = vals[0].reshape(*new_res)
    # n2 = torch.ones_like(s)
    # inside = (slice(1, -1),)*n.ndim
    # n2[inside] = s[inside]
    return s


def get_pts_sdf(sdf, nrays, width):
    h = width/(sdf.shape[0]-1)
    pts = width*torch.rand(nrays, 3)

    # TODO: Make sure the sdf values have the right scale for proper tracing
    vol = Grid(h*sdf, h)

    dist, distx = vol.GetLinear(pts)
    dnorm = torch.norm(distx, dim=-1, keepdim=True)
    vel = distx / dnorm

    pos = pts - dist[:, None] * vel
    pos -= h * distx / 10

    mask = dist > -1e-6
    eps = 1 / 10
    dist = dist[mask]
    for i in range(1000):
        if not torch.any(mask):
            print('all ray success')
            break
        pos[mask] -= eps * dist[:, None] * vel[mask] / (i+1)
        dist, distx = vol.GetLinear(pos[mask])
        mask[mask] = dist > -1e-6

    return pos, -vel


def get_opp_pts(sdf, pts, v, width):
    h = width/(sdf.shape[0]-1)
    vol = Grid(sdf, h)

    dist, distx = vol.GetLinear(pts)

    pos = pts.clone()
    mask = dist < 0
    for i in range(sdf.shape[0]*3):
        if not torch.any(mask):
            print('all ray match success')
            break
        pos[mask] += h * v[mask] / 2
        dist, distx = vol.GetLinear(pos[mask])
        mask[mask] = dist < 0

    return pos
