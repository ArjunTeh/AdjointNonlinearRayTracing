import torch
import enoki
# float
from enoki.dynamic import Float32 as FloatS, Vector3f as Vector3fS
from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD
from enoki.cuda import Float32 as FloatC, Vector3f as Vector3fC

# double
# from enoki.dynamic import Float64 as FloatS, Vector3d as Vector3fS
# from enoki.cuda_autodiff import Float64 as FloatD, Vector3d as Vector3fD
# from enoki.cuda import Float64 as FloatC, Vector3d as Vector3fC

import drrt


class ADTracerC(torch.autograd.Function):

    @staticmethod
    def forward(ctx, rif, x, v, h, ds):
        ctx.shape = rif.shape
        ctx.rif = FloatD(rif.flatten())
        ctx.x = Vector3fD(x)
        ctx.v = Vector3fD(v)
        ctx.h = h
        ctx.ds = ds

        enoki.set_requires_gradient(ctx.rif, rif.requires_grad)
        enoki.set_requires_gradient(ctx.x, x.requires_grad)
        enoki.set_requires_gradient(ctx.v, v.requires_grad)

        trace_fun = drrt.TracerD()
        ctx.outx, ctx.outv = trace_fun.trace(ctx.rif,
                                             ctx.shape,
                                             ctx.x,
                                             ctx.v,
                                             h,
                                             ds)

        out_torch = (ctx.outx.torch(), ctx.outv.torch())
        # enoki.cuda_malloc_trim()

        return out_torch

    @staticmethod
    def backward(ctx, grad_x, grad_v):

        enoki.set_gradient(ctx.outx, Vector3fC(grad_x))
        enoki.set_gradient(ctx.outv, Vector3fC(grad_v))

        FloatD.backward()

        # h and ds are not differentiable!
        result = (enoki.gradient(ctx.rif).torch().reshape(*ctx.shape)
                  if enoki.requires_gradient(ctx.rif) else None,
                  enoki.gradient(ctx.x).torch()
                  if enoki.requires_gradient(ctx.x) else None,
                  enoki.gradient(ctx.v).torch()
                  if enoki.requires_gradient(ctx.v) else None,
                  None,
                  None)

        # cleanup
        del ctx.outx, ctx.outv, ctx.x, ctx.v, ctx.rif, ctx.h, ctx.ds
        enoki.cuda_malloc_trim()

        return result


class ADTracerS(torch.autograd.Function):

    @staticmethod
    def forward(ctx, rif, x, v, h, ds):
        ctx.shape = rif.shape
        ctx.rif = FloatS(rif.flatten())
        ctx.x = Vector3fS(x)
        ctx.v = Vector3fS(v)
        ctx.h = h
        ctx.ds = ds

        enoki.set_requires_gradient(ctx.rif, rif.requires_grad)
        enoki.set_requires_gradient(ctx.x, x.requires_grad)
        enoki.set_requires_gradient(ctx.v, v.requires_grad)

        trace_fun = drrt.TracerDS()
        ctx.outx, ctx.outv = trace_fun.trace(ctx.rif,
                                             ctx.shape,
                                             ctx.x,
                                             ctx.v,
                                             h,
                                             ds)

        out_torch = (ctx.outx.torch(), ctx.outv.torch())
        # enoki.cuda_malloc_trim()

        return out_torch

    @staticmethod
    def backward(ctx, grad_x, grad_v):

        enoki.set_gradient(ctx.outx, Vector3fC(grad_x))
        enoki.set_gradient(ctx.outv, Vector3fC(grad_v))

        FloatS.backward()

        # h and ds are not differentiable!
        result = (enoki.gradient(ctx.rif).torch().reshape(*ctx.shape)
                  if enoki.requires_gradient(ctx.rif) else None,
                  enoki.gradient(ctx.x).torch()
                  if enoki.requires_gradient(ctx.x) else None,
                  enoki.gradient(ctx.v).torch()
                  if enoki.requires_gradient(ctx.v) else None,
                  None,
                  None)

        # cleanup
        del ctx.outx, ctx.outv, ctx.x, ctx.v, ctx.rif, ctx.h, ctx.ds
        # enoki.cuda_malloc_trim()

        return result


class ADPlaneTracerC(torch.autograd.Function):

    @staticmethod
    def forward(ctx, rif, x, v, sp, sn, h, ds):
        ctx.shape = rif.shape
        ctx.rif = FloatD(rif.flatten())
        ctx.x = Vector3fD(x)
        ctx.v = Vector3fD(v)
        ctx.sp = Vector3fD(sp)
        ctx.sn = Vector3fD(sn)
        ctx.h = h
        ctx.ds = ds

        enoki.set_requires_gradient(ctx.rif, rif.requires_grad)
        enoki.set_requires_gradient(ctx.x, x.requires_grad)
        enoki.set_requires_gradient(ctx.v, v.requires_grad)

        trace_fun = drrt.TracerD()
        ctx.outx, ctx.outv = trace_fun.trace_plane(ctx.rif,
                                                   ctx.shape,
                                                   ctx.x,
                                                   ctx.v,
                                                   ctx.sp,
                                                   ctx.sn,
                                                   h,
                                                   ds)

        out_torch = (ctx.outx.torch(), ctx.outv.torch())
        # enoki.cuda_malloc_trim()

        return out_torch

    @staticmethod
    def backward(ctx, grad_x, grad_v):

        enoki.set_gradient(ctx.outx, Vector3fC(grad_x))
        enoki.set_gradient(ctx.outv, Vector3fC(grad_v))

        FloatD.backward()

        # h and ds are not differentiable!
        result = (enoki.gradient(ctx.rif).torch().reshape(*ctx.shape)
                  if enoki.requires_gradient(ctx.rif) else None,
                  enoki.gradient(ctx.x).torch()
                  if enoki.requires_gradient(ctx.x) else None,
                  enoki.gradient(ctx.v).torch()
                  if enoki.requires_gradient(ctx.v) else None,
                  None,
                  None,
                  None,
                  None)

        # cleanup
        del ctx.outx, ctx.outv, ctx.x, ctx.v, ctx.sp, ctx.sn, ctx.rif, ctx.h, ctx.ds
        enoki.cuda_malloc_trim()

        return result


class ADSDFTracerC(torch.autograd.Function):

    @staticmethod
    def forward(ctx, rif, sdf, x, v, h, ds):
        ctx.shape = rif.shape
        ctx.rif = FloatD(rif.flatten())
        ctx.sdf = FloatD(sdf.flatten())
        ctx.x = Vector3fD(x)
        ctx.v = Vector3fD(v)
        ctx.h = h
        ctx.ds = ds

        enoki.set_requires_gradient(ctx.rif, rif.requires_grad)
        enoki.set_requires_gradient(ctx.x, x.requires_grad)
        enoki.set_requires_gradient(ctx.v, v.requires_grad)

        trace_fun = drrt.TracerD()
        ctx.outx, ctx.outv = trace_fun.trace_sdf(ctx.rif,
                                                 ctx.sdf,
                                                 ctx.shape,
                                                 ctx.x,
                                                 ctx.v,
                                                 h,
                                                 ds)

        out_torch = (ctx.outx.torch(), ctx.outv.torch())
        # enoki.cuda_malloc_trim()

        return out_torch

    @staticmethod
    def backward(ctx, grad_x, grad_v):

        enoki.set_gradient(ctx.outx, Vector3fC(grad_x))
        enoki.set_gradient(ctx.outv, Vector3fC(grad_v))

        FloatD.backward()

        # h and ds are not differentiable!
        result = (enoki.gradient(ctx.rif).torch().reshape(*ctx.shape)
                  if enoki.requires_gradient(ctx.rif) else None,
                  None,
                  enoki.gradient(ctx.x).torch()
                  if enoki.requires_gradient(ctx.x) else None,
                  enoki.gradient(ctx.v).torch()
                  if enoki.requires_gradient(ctx.v) else None,
                  None,
                  None)

        # cleanup
        del ctx.outx, ctx.sdf, ctx.outv, ctx.x, ctx.v, ctx.rif, ctx.h, ctx.ds
        enoki.cuda_malloc_trim()

        return result


class ADCableTracerC(torch.autograd.Function):

    @staticmethod
    def forward(ctx, rif, radius, length, x, v, sp, ds):
        ctx.radius = radius
        ctx.length = length
        ctx.rif = FloatD(rif)
        ctx.x = Vector3fD(x)
        ctx.v = Vector3fD(v)
        ctx.sp = Vector3fD(sp)
        ctx.ds = ds

        enoki.set_requires_gradient(ctx.rif, rif.requires_grad)
        enoki.set_requires_gradient(ctx.x, x.requires_grad)
        enoki.set_requires_gradient(ctx.v, v.requires_grad)

        trace_fun = drrt.TracerD()
        ctx.outx, ctx.outv, ctx.outdist2 = trace_fun.trace_cable(ctx.rif,
                                                                ctx.radius,
                                                                ctx.length,
                                                                ctx.x,
                                                                ctx.v,
                                                                ctx.sp,
                                                                ds)

        out_torch = (ctx.outx.torch(), ctx.outv.torch(), ctx.outdist2.torch())
        # enoki.cuda_malloc_trim()

        return out_torch

    @staticmethod
    def backward(ctx, grad_x, grad_v, grad_dist):

        enoki.set_gradient(ctx.outx, Vector3fC(grad_x))
        enoki.set_gradient(ctx.outv, Vector3fC(grad_v))

        FloatD.backward()

        # h and ds are not differentiable!
        result = (enoki.gradient(ctx.rif).torch()
                  if enoki.requires_gradient(ctx.rif) else None,
                  None,
                  None,
                  enoki.gradient(ctx.x).torch()
                  if enoki.requires_gradient(ctx.x) else None,
                  enoki.gradient(ctx.v).torch()
                  if enoki.requires_gradient(ctx.v) else None,
                  None,
                  None)

        # cleanup
        del ctx.outx, ctx.outv, ctx.x, ctx.v, ctx.rif, ctx.ds
        enoki.cuda_malloc_trim()

        return result


class BackTracerC(torch.autograd.Function):

    @staticmethod
    def forward(ctx, rif, x, v, h, ds):
        ctx.shape = rif.shape
        ctx.rif = FloatC(rif.flatten())
        ctx.x = Vector3fC(x.detach())
        ctx.v = Vector3fC(v.detach())
        ctx.h = h
        ctx.ds = ds

        trace_fun = drrt.TracerC()
        ctx.outx, ctx.outv = trace_fun.trace(ctx.rif,
                                             ctx.shape,
                                             ctx.x,
                                             ctx.v,
                                             h,
                                             ds)

        out_torch = (ctx.outx.torch(), ctx.outv.torch())
        enoki.cuda_malloc_trim()

        return out_torch

    @staticmethod
    def backward(ctx, grad_x, grad_v):

        grad_x_ek = Vector3fC(grad_x)
        grad_v_ek = Vector3fC(grad_v)

        trace_fun = drrt.TracerC()
        rif_result = trace_fun.backtrace(ctx.rif,
                                         ctx.shape,
                                         ctx.outx,
                                         ctx.outv,
                                         grad_x_ek,
                                         grad_v_ek,
                                         ctx.h,
                                         ctx.ds)
        drif = rif_result.torch().reshape(*ctx.shape)

        return drif, None, None, None, None


class BackPlaneTracerC(torch.autograd.Function):

    @staticmethod
    def forward(ctx, rif, x, v, sp, sn, h, ds):
        ctx.shape = rif.shape
        ctx.rif = FloatC(rif.flatten())
        ctx.x = Vector3fC(x.detach())
        ctx.v = Vector3fC(v.detach())
        ctx.sp = Vector3fC(sp.detach())
        ctx.sn = Vector3fC(sn.detach())
        ctx.h = h
        ctx.ds = ds

        trace_fun = drrt.TracerC()
        ctx.outx, ctx.outv, ctx.outmask = trace_fun.trace_pln(ctx.rif,
                                                              ctx.shape,
                                                              ctx.x,
                                                              ctx.v,
                                                              ctx.sp,
                                                              ctx.sn,
                                                              h,
                                                              ds)

        out_torch = (ctx.outx.torch(), ctx.outv.torch(), ctx.outmask.torch().to(torch.bool))
        enoki.cuda_malloc_trim()

        return out_torch

    @staticmethod
    def backward(ctx, grad_x, grad_v, outmask):

        if outmask is not None:
            grad_x[outmask] = 0
        grad_x_ek = Vector3fC(grad_x)
        grad_v_ek = Vector3fC(grad_v)


        trace_fun = drrt.TracerC()
        rif_result = trace_fun.backtrace(ctx.rif,
                                         ctx.shape,
                                         ctx.outx,
                                         ctx.outv,
                                         grad_x_ek,
                                         grad_v_ek,
                                         ctx.h,
                                         ctx.ds)
        drif = rif_result.torch().reshape(*ctx.shape)

        return drif, None, None, None, None, None, None


class BackTargetTracerC(torch.autograd.Function):

    @staticmethod
    def forward(ctx, rif, x, v, sp, h, ds):
        ctx.shape = rif.shape
        ctx.rif = FloatC(rif.flatten())
        ctx.x = Vector3fC(x.detach())
        ctx.v = Vector3fC(v.detach())
        ctx.sp = Vector3fC(sp.detach())
        ctx.h = h
        ctx.ds = ds

        trace_fun = drrt.TracerC()
        ctx.outx, ctx.outv, ctx.outdist2 = trace_fun.trace_target(ctx.rif,
                                                                  ctx.shape,
                                                                  ctx.x,
                                                                  ctx.v,
                                                                  ctx.sp,
                                                                  h,
                                                                  ds)

        out_torch = (ctx.outx.torch(), ctx.outv.torch(), ctx.outdist2.torch())
        enoki.cuda_malloc_trim()

        return out_torch

    @staticmethod
    def backward(ctx, grad_x, grad_v, outdist):

        grad_x_ek = Vector3fC(grad_x)
        grad_v_ek = Vector3fC(grad_v)

        trace_fun = drrt.TracerC()
        rif_result = trace_fun.backtrace(ctx.rif,
                                         ctx.shape,
                                         ctx.outx,
                                         ctx.outv,
                                         grad_x_ek,
                                         grad_v_ek,
                                         ctx.h,
                                         ctx.ds)
        drif = rif_result.torch().reshape(*ctx.shape)

        return drif, None, None, None, None, None


class BackSDFTracerC(torch.autograd.Function):

    @staticmethod
    def forward(ctx, rif, sdf, x, v, h, ds):
        ctx.shape = rif.shape
        ctx.rif = FloatC(rif.flatten())
        ctx.sdf = FloatC(sdf.flatten())
        ctx.x = Vector3fC(x.detach())
        ctx.v = Vector3fC(v.detach())
        ctx.h = h
        ctx.ds = ds

        trace_fun = drrt.TracerC()
        ctx.outx, ctx.outv = trace_fun.trace_sdf(ctx.rif,
                                                 ctx.sdf,
                                                 ctx.shape,
                                                 ctx.x,
                                                 ctx.v,
                                                 h,
                                                 ds)

        out_torch = (ctx.outx.torch(), ctx.outv.torch())
        enoki.cuda_malloc_trim()

        return out_torch

    @staticmethod
    def backward(ctx, grad_x, grad_v):

        grad_x_ek = Vector3fC(grad_x)
        grad_v_ek = Vector3fC(grad_v)

        trace_fun = drrt.TracerC()
        rif_result = trace_fun.backtrace_sdf(ctx.rif,
                                             ctx.sdf,
                                             ctx.shape,
                                             ctx.outx,
                                             ctx.outv,
                                             grad_x_ek,
                                             grad_v_ek,
                                             ctx.h,
                                             ctx.ds)
        drif = rif_result.torch().reshape(*ctx.shape)

        return drif, None, None, None, None, None


class BackCableTracerC(torch.autograd.Function):

    @staticmethod
    def forward(ctx, rif, radius, length, x, v, sp, ds):

        ctx.radius = radius
        ctx.length = length
        ctx.rif = FloatC(rif.flatten())
        ctx.x = Vector3fC(x.detach())
        ctx.v = Vector3fC(v.detach())
        ctx.sp = Vector3fC(sp.detach())
        ctx.ds = ds

        trace_fun = drrt.TracerC()
        ctx.outx, ctx.outv, ctx.outdist2 = trace_fun.trace_cable(ctx.rif,
                                                                 ctx.radius,
                                                                 ctx.length,
                                                                 ctx.x,
                                                                 ctx.v,
                                                                 ctx.sp,
                                                                 ctx.ds)

        out_torch = (ctx.outx.torch(), ctx.outv.torch(), ctx.outdist2.torch())
        enoki.cuda_malloc_trim()

        return out_torch

    @staticmethod
    def backward(ctx, grad_x, grad_v, outdist):

        grad_x_ek = Vector3fC(grad_x)
        grad_v_ek = Vector3fC(grad_v)

        trace_fun = drrt.TracerC()
        rif_result = trace_fun.backtrace_cable(ctx.rif,
                                               ctx.radius,
                                               ctx.length,
                                               ctx.outx,
                                               ctx.outv,
                                               grad_x_ek,
                                               grad_v_ek,
                                               ctx.ds)
        drif = rif_result.torch()

        return drif, None, None, None, None, None, None
