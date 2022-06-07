import torch
import torch.optim as optim
import grid
from tqdm.auto import tqdm


def upres_scene(n, res):
    # double the scene resolution
    upres = [res for i in n.shape]
    return grid.upres_volume(n.detach().to(torch.double), upres).to(n.dtype)


def reload_opto(old_o, n, lr):
    # assume that there is only one value to upsample
    ogroup, state = None, None
    for group in old_o.param_groups:
        ogroup = group
        # print('beta', ogroup['betas'])
        for p in group['params']:
            if len(old_o.state[p]) == 0:
                # The optimizer hasn't even started yet
                continue
            state = dict()
            ostate = old_o.state[p]
            state['step'] = ostate['step']
            state['exp_avg'] = upres_scene(ostate['exp_avg'], n.shape[0])
            # print('range:', ostate['exp_avg'].max(), ostate['exp_avg'].min())
            # print('rangeU:', state['exp_avg'].max(), state['exp_avg'].min())
            state['exp_avg_sq'] = upres_scene(ostate['exp_avg_sq'], n.shape[0])

    opto = optim.Adam([n], lr=lr)
    for group in opto.param_groups:
        if ogroup is not None:
            group['betas'] = ogroup['betas']
            group['lr'] = ogroup['lr']
            group['weight_decay'] = ogroup['weight_decay']
            group['eps'] = ogroup['eps']
        for p in group['params']:
            if state is not None:
                opto.state[p] = state
    return opto


def multires_opt(func, eta, iterations, res_list, log_func=None, lr=1e-3, statename='result'):
    n = eta.clone()
    n.requires_grad = True

    opto = optim.Adam([n], lr=lr)
    iteration_count = 0
    loss_hist = []
    for res_iter in tqdm(range(len(res_list))):

        mask = torch.ones_like(n, dtype=bool, requires_grad=False)
        mask[1:-1, 1:-1, 1:-1] = 0
        for j in tqdm(range(iterations*(res_iter+1))):
            opto.zero_grad()

            loss = func(n)
            loss.backward()

            with torch.no_grad():
                log_func(iteration_count, n)
                n.grad[mask] = 0

            opto.step()

            with torch.no_grad():
                n.clamp_(min=1)
                loss_hist.append(loss.item())

            iteration_count += 1
        
        with torch.no_grad():
            torch.save({
                'rif': n,
                'opto_state_dict': opto.state_dict(),
                'loss_hist': torch.tensor(loss_hist)
            }, statename)
            if res_iter < len(res_list)-1:
                n = upres_scene(n, res_list[res_iter+1])
                n.requires_grad = True
                opto = reload_opto(opto, n, (0.5**res_iter)*lr)

    return n, loss_hist