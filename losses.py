import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
# from helpers import get_device
import numpy as np


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - \
        torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                        keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                   keepdim=True) + lnB + lnB_uni
    return kl


def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum(
        (y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(torch.tensor(
        1.0, dtype=torch.float32), torch.tensor(epoch_num / annealing_step, dtype=torch.float32))

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * \
        kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, p, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    #print(y.shape)
    #print(p)
    A = p * torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(torch.tensor(
        1.0, dtype=torch.float32), torch.tensor(epoch_num / annealing_step, dtype=torch.float32))

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * \
        kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(mse_loss(target, alpha, epoch_num,
                               num_classes, annealing_step, device=device))
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(edl_loss(torch.log, target, alpha,
                               epoch_num, num_classes, annealing_step, device))
    return loss


def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step, p, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(edl_loss(torch.digamma, target, alpha,
                               epoch_num, num_classes, annealing_step, p, device))
    return loss




def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5*(a1-1)/b1 * (v2*torch.square(mu2-mu1))  \
        + 0.5*v2/v1  \
        - 0.5*torch.log(torch.abs(v2)/torch.abs(v1))  \
        - 0.5 + a2*torch.log(b1/b2)  \
        - (torch.lgamma(a1) - torch.lgamma(a2))  \
        + (a1 - a2)*torch.digamma(a1)  \
        - (b1 - b2)*a1/b1
    return KL

def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2*beta*(1+v)
    nll = 0.5*torch.log(np.pi/v) - alpha*torch.log(twoBlambda)  + (alpha+0.5) * torch.log(v*(y-gamma)**2 + twoBlambda)  + torch.lgamma(alpha) - torch.lgamma(alpha+0.5)
    # Tobi: Changed here to abs
    nll = torch.abs(nll)
    #
    s = torch.sum(nll,dim=1, keepdim=True)
    s = torch.mean(s)
    return s
    #return tf.reduce_mean(nll) if reduce else nll

def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
    # error = tf.stop_gradient(tf.abs(y-gamma))
    error = torch.abs(y-gamma)
    
    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
        reg = error*kl
    else:
        evi = 2*v+(alpha)
        reg = error*evi
        
    s = torch.sum(reg,dim=1, keepdim=True)
    s = torch.mean(s)
    return s
    #return tf.reduce_mean(reg) if reduce else reg

def EvidentialRegression(y_true, gamma, v, alpha, beta, coeff=1.0):
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)
    return loss_nll + coeff * loss_reg


class RegularizationLoss(Module):
    """
    ## Regularization loss
    $$L_{Reg} = \mathop{KL} \Big(p_n \Vert p_G(\lambda_p) \Big)$$
    $\mathop{KL}$ is the [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).
    $p_G$ is the [Geometric distribution](https://en.wikipedia.org/wiki/Geometric_distribution) parameterized by
    $\lambda_p$. *$\lambda_p$ has nothing to do with $\lambda_n$; we are just sticking to same notation as the paper*.
    $$Pr_{p_G(\lambda_p)}(X = k) = (1 - \lambda_p)^k \lambda_p$$.
    The regularization loss biases the network towards taking $\frac{1}{\lambda_p}$ steps and incentivies non-zero probabilities
    for all steps; i.e. promotes exploration.
    """

    def __init__(self, lambda_p: float, max_steps: int = 1):
        """
        * `lambda_p` is $\lambda_p$ - the success probability of geometric distribution
        * `max_steps` is the highest $N$; we use this to pre-compute $p_G(\lambda_p)$
        """
        super().__init__()

        # Empty vector to calculate $p_G(\lambda_p)$
        p_g = torch.zeros((max_steps,))
        # $(1 - \lambda_p)^k$
        not_halted = 1.
        # Iterate upto `max_steps`
        for k in range(max_steps):
            # $$Pr_{p_G(\lambda_p)}(X = k) = (1 - \lambda_p)^k \lambda_p$$
            p_g[k] = not_halted * lambda_p
            # Update $(1 - \lambda_p)^k$
            not_halted = not_halted * (1 - lambda_p)

        # Save $Pr_{p_G(\lambda_p)}$
        self.p_g = nn.Parameter(p_g, requires_grad=False)

        # KL-divergence loss
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, p: torch.Tensor, cuda=False):
        """
        * `p` is $p_1 \dots p_N$ in a tensor of shape `[N, batch_size]`
        """
        # Transpose `p` to `[batch_size, N]`
        p = p.transpose(0, 1)
        # Get $Pr_{p_G(\lambda_p)}$ upto $N$ and expand it across the batch dimension
        p_g = self.p_g[None, :p.shape[1]].expand_as(p)
        if cuda:
            p_g = p_g.cuda()
            p = p.cuda()
        # Calculate the KL-divergence.
        # *The [PyTorch KL-divergence](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)
        # implementation accepts log probabilities.*
        return self.kl_div(p.log(), p_g)

