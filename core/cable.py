import torch


class Cable:

    def __init__(self, rif, radius, length):
        self.rif = rif
        self.res = rif.size()
        self.radius = radius
        self.length = length
        self.h = radius / (rif.shape[0] - 1)
        self.device = rif.device

    def check_input(self, x):
        if x.device != self.device:
            raise ValueError("input on device: {}, grid on device: {}"
                             .format(x.device, self.device))

    def bounds(self, x):
        r = torch.norm(x[:, [0, 2]])
        l = torch.abs(x[:, 1] - (self.length/2))

        return (r < self.radius) & (l < (self.length/2))

    def render(self, res):
        if len(res) != 3:
            raise ValueError("res must be of dimension 3")

        X = torch.meshgrid([
            torch.linspace(0, 2*self.radius, res[0]),
            torch.linspace(0, self.length, res[1]),
            torch.linspace(0, 2*self.radius, res[2])
        ])
        pos = torch.stack([x.flatten() for x in X], dim=-1)
        
        n, nx = self.GetLinear(pos)
        return n.reshape(res)

    def render2(self, res):
        if type(res) == int:
            res = [res, res]

        if len(res) != 2:
            raise ValueError("res must be int or of length 2")

        X = torch.meshgrid([
            torch.linspace(0, 2*self.radius, res[0]),
            torch.linspace(0, self.length, res[1])
        ])

        pos = torch.stack([
            X[0].flatten(),
            X[1].flatten(),
            torch.ones(X[0].numel())*self.radius
        ], dim=-1)

        n, nx = self.GetLinear(pos)
        return n.reshape(res)

    def RenderGradient(self, linear=False):
        dev = self.rif.device
        idx = torch.meshgrid(*[self.h*torch.arange(r, device=dev) for r in self.res])
        z = torch.stack([x.flatten() for x in idx], dim=-1)
        f, fx = self.GetLinear(z)
        return fx.reshape(*self.res, self.rif.ndim)

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

    # Bi/Trilinear interpolation implementation
    def GetLinear(self, x):
        self.check_input(x)

        xn = x.clone() - self.radius
        xn[:, 1] = 0

        r = torch.norm(xn, dim=-1)
        rn = (r / self.h)

        x0 = torch.floor(rn).long()
        w0 = torch.clip(rn - x0, 0, 1)

        idx = [x0, x0+1]
        weights = [1-w0, w0]

        capped = torch.stack([torch.clip(x, 0, self.res[0]-1) for x in idx])
        fi = self.rif[capped]
        wi = torch.stack(weights)

        f = torch.sum(fi * wi, dim=0)

        rgrad = fi[1] - fi[0]
        rx = xn / r[:, None]
        rx[r < 1e-6] = 0

        fx = rgrad[:, None] * rx

        return f, fx / self.h


def upres_volume(n, new_res):
    nvox = torch.clip(torch.tensor(n.shape[0]-1), min=1)
    gt = Cable(n, 1 / nvox)
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
