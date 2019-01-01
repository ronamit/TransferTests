from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import torch
import torch.optim as optim
from torch.autograd import Variable

matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

# -------------------------------------------------------------------------------------------
#  Define  parameters
# -------------------------------------------------------------------------------------------

n_tasks = 3

m_samples = 50 # samples per task

dim = 2

hyper_prior_mean = [0.0, 0.0]
hyper_prior_sigmas = [1.0, 1.0]
alphas_sigmas = [0.3, 0.3]
data_sigmas = [0.1, 0.1]

# -------------------------------------------------------------------------------------------
#  Create data
# -------------------------------------------------------------------------------------------


# for each scenario

# sample psi from hyper-prior

psi = np.random.multivariate_normal(
    mean=hyper_prior_mean,
    cov=np.diag(hyper_prior_sigmas),
    size=1)
psi = psi[0]

# sample tasks' alphas from conditional prior
alphas = np.random.multivariate_normal(
    mean=psi,
    cov=np.diag(alphas_sigmas),
    size=n_tasks)



# sample data from true posterior
data_set = []
for i_task in range(n_tasks):
    task_data = np.random.multivariate_normal(
        mean=alphas[i_task],
        cov=np.diag(data_sigmas),
        size=m_samples)
    task_data = torch.from_numpy(task_data).type(torch.FloatTensor).to(device)
    data_set.append(task_data)


alphas = torch.from_numpy(alphas).to(device)
# -------------------------------------------------------------------------------------------
#  Meta-learning
# -------------------------------------------------------------------------------------------

# Define theta (learned mean parameters of the approx. posterior over shared latent variable psi):
theta = torch.randn(dim, device=device, requires_grad=True)

# Define phi variables (learned mean parameters of the approx. posteriors over task latent variables alpha)
phis = torch.randn(n_tasks, dim, device=device,  requires_grad=True)
print(phis)


# create your optimizer
learning_rate = 1e-1
optimizer = optim.Adam([theta, phis], lr=learning_rate)

n_epochs = 800
batch_m = 128 # number of samples from each task to take to batch
batch_m = min(batch_m, m_samples)

for i_epoch in range(n_epochs):

    empirical_loss = torch.zeros(1)

    for i_task in range(n_tasks):
        # Sample data batch for current task:
        batch_inds = np.random.choice(m_samples, batch_m, replace=False)
        task_data = data_set[i_task][batch_inds]

        # # Re-Parametrization:
        # w_sigma = torch.exp(w_log_sigma[b_task])
        # epsilon = Variable(torch.randn(n_dim).cuda(), requires_grad=False)
        # w = w_mu[b_task] + w_sigma * epsilon

        w = phis[i_task]

        # calculate empirical loss estimate:
        empirical_loss += (w - task_data).pow(2).mean()


# -------------------------------------------------------------------------------------------
# Learning a new task
# -------------------------------------------------------------------------------------------
