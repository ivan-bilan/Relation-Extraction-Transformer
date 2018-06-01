"""
Utility functions for torch.
"""

import torch
from torch import nn, optim
from torch.optim import Optimizer


### class
class MyAdagrad(Optimizer):
    """My modification of the Adagrad optimizer that allows to specify an initial
    accumulater value. This mimics the behavior of the default Adagrad implementation
    in Tensorflow. The default PyTorch Adagrad uses 0 for initial acculmulator value.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        init_accu_value (float, optional): initial accumulator value.
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, init_accu_value=0.1, weight_decay=0):
        defaults = dict(lr=lr, lr_decay=lr_decay, init_accu_value=init_accu_value, \
                weight_decay=weight_decay)
        super(MyAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.ones(p.data.size()).type_as(p.data) *\
                        init_accu_value

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients ")
                    grad = grad.add(group['weight_decay'], p.data)

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                if p.grad.data.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = torch.Size([x for x in grad.size()])

                    def make_sparse(values):
                        constructor = type(p.grad.data)
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor()
                        return constructor(grad_indices, values, size)
                    state['sum'].add_(make_sparse(grad_values.pow(2)))
                    std = state['sum']._sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(1e-10)
                    p.data.add_(-clr, make_sparse(grad_values / std_values))
                else:
                    state['sum'].addcmul_(1, grad, grad)
                    std = state['sum'].sqrt().add_(1e-10)
                    p.data.addcdiv_(-clr, grad, std)

        return loss


class NAdam(Optimizer):
    """
    Implements Nesterov-accelerated Adam algorithm according to Keras.
    parameter name alias in different algorithms
    NAdam                           Keras                         054_report
    exp_avg                         m_t                            m_t
    exp_avg_prime              \prime{m}_t              \prime{m}_t
    exp_avg_bar                  \bar{m}_t                  \bar{m}_t
    exp_avg_sq                    v_t                             n_t
    exp_avg_sq_prime         \prime{v}_t               \prime{n}_t
    beta1                              beta_1                       \mu
    beta2                              beta_2                       \v=0.999
    It has been proposed in `Incorporating Nesterov Momentum into Adam`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0),
            but not used in NAdam
        schedule_decay (float, optional): coefficients used for computing
            moment schedule (default: 0.004)
    .. _Incorporating Nesterov Momentum into Adam
        http://cs229.stanford.edu/proj2015/054_report.pdf
    .. _On the importance of initialization and momentum in deep learning
        http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, schedule_decay=0.004):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, schedule_decay=schedule_decay)
        super(NAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NAdam, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('NAdam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # \mu^{t}
                    state['m_schedule'] = 1.

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']

                schedule_decay = group['schedule_decay']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # calculate the momentum cache \mu^{t} and \mu^{t+1}
                momentum_cache_t = beta1 * (
                    1. - 0.5 * (pow(0.96, state['step'] * schedule_decay)))
                momentum_cache_t_1 = beta1 * (
                    1. - 0.5 * (pow(0.96, (state['step'] + 1) * schedule_decay)))
                m_schedule_new = state['m_schedule'] * momentum_cache_t
                m_schedule_next = state['m_schedule'] * momentum_cache_t * momentum_cache_t_1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                g_prime = torch.div(grad, 1. - m_schedule_new)
                exp_avg_prime = torch.div(exp_avg, 1. - m_schedule_next)
                exp_avg_sq_prime = torch.div(exp_avg_sq, 1. - pow(beta2, state['step']))

                exp_avg_bar = torch.add((1. - momentum_cache_t) * g_prime,
                                        momentum_cache_t_1, exp_avg_prime)

                denom = exp_avg_sq_prime.sqrt().add_(group['eps'])

                step_size = group['lr']

                p.data.addcdiv_(-step_size, exp_avg_bar, denom)

                return loss


class NoamOpt:
    "Optim wrapper that implements rate."

    # TODO: this leads to errors

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


### torch specific functions
def get_optimizer(name, parameters, lr):
    if name == 'sgd':
        # TODO: test momentum and weight_decay
        # 1e-07 decay? , decay is like l2, try without!!!  # weight_decay=1e-6
        return torch.optim.SGD(parameters, lr=lr)  # bad results: weight_decay=1e-07, , momentum=0.9, nesterov=True
    elif name == 'nadam':
        return NAdam(parameters, lr=lr)
    elif name == 'asgd':
        return torch.optim.ASGD(parameters, lr=lr)
    elif name in ['adagrad', 'myadagrad']:
        # use new adagrad to allow for init accumulator value
        return MyAdagrad(parameters, lr=lr, init_accu_value=0.1)
    elif name == 'adam':
        return torch.optim.Adam(parameters, betas=(0.9, 0.98), lr=lr, eps=1e-9)  # , amsgrad=True
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr)
    elif name == "noopt_adam":
        # TODO: doesn't seem to work properly
        # this comes from http://nlp.seas.harvard.edu/2018/04/03/attention.html
        return NoamOpt(360, 1, 400, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9))
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def flatten_indices(seq_lens, width):
    flat = []
    for i, l in enumerate(seq_lens):
        for j in range(l):
            flat.append(i * width + j)
    return flat


def set_cuda(var, cuda):
    if cuda:
        return var.cuda()
    return var


def keep_partial_grad(grad, topk):
    """
    Keep only the topk rows of grads.
    """
    assert topk < grad.size(0)
    grad.data[topk:].zero_()
    return grad


### model IO
def save(model, optimizer, opt, filename):
    params = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': opt
    }
    try:
        torch.save(params, filename)
    except BaseException:
        print("[ Warning: model saving failed. ]")


def load(model, optimizer, filename):
    try:
        dump = torch.load(filename)
    except BaseException:
        print("[ Fail: model loading failed. ]")
    if model is not None:
        model.load_state_dict(dump['model'])
    if optimizer is not None:
        optimizer.load_state_dict(dump['optimizer'])
    opt = dump['config']
    return model, optimizer, opt


def load_config(filename):
    try:
        dump = torch.load(filename)
    except BaseException:
        print("[ Fail: model loading failed. ]")
    return dump['config']
