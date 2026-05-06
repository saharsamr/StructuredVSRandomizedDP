import torch
from torch.optim import Adam

class AdamWDPBC(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, dp_bias_correction=0):
        self.dp_bias_correction = dp_bias_correction
        super(AdamWDPBC, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, use SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                beta1, beta2 = group['betas']

                state['step'] += 1

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group['amsgrad']:
                    # Maintains the maximum of all second moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max for normalizing the step size instead of the second moment estimate
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    #denom = (exp_avg_sq.sqrt() + group['eps'])  # This is the original

                    # Correct DP bias
                    denom = (torch.max(exp_avg_sq - self.dp_bias_correction, torch.zeros(exp_avg_sq.shape, device=exp_avg_sq.device) + 5e-8)).sqrt()   

                step_size = group['lr']

                # Perform parameter update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
