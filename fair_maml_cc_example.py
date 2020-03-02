"""
This code is adapted from: https://github.com/dragen1860/MAML-Pytorch.
"""

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim

import numpy as np
from copy import deepcopy

from utils import *
import time

from data.communites_and_crime.get_data import get_splits, get_and_preprocess_communities_and_crime_data

import os

np.random.seed(112233)
torch.manual_seed(112233)

NUM_TESTING_TASKS = 5

tasks, cols = get_and_preprocess_communities_and_crime_data()
lo, li, tasks, training_batches, testing_batches = get_splits(tasks,new=True)

### Setup run paremeters ###
EPOCHS = 2000
METRIC = "DP"

GAMMA_RANGE = [0, 4, 5, 6]
fairness_reg = disparate_impact_reg
fairness_notion = disparate_impact

print ("Sweeping over GAMMA range {} with {} regularization.".format(GAMMA_RANGE, METRIC))
print ("---------")
######

def gen_batch(testing=False):
    if testing:
        return testing_batches[0],testing_batches[1],testing_batches[2],testing_batches[3],testing_batches[4],testing_batches[5]
    else:
        batch = training_batches[np.random.choice(len(training_batches))]    
        return batch[0],batch[1],batch[2],batch[3],batch[4],batch[5]

class Learner(nn.Module):
    def __init__(self, config):
        super(Learner, self).__init__()

        self.config = config
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name is 'linear':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'relu':
                continue
            else:
                raise NotImplementedError

    def forward(self, x, vars=None, bn_training=True):
        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2

            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            else:
                raise NotImplementedError

        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return x

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()
    
    def parameters(self):
        return self.vars

class Meta(nn.Module):
    def __init__(self, config, tasks, gamma=10, update_lr=0.4, meta_lr=1e-3, 
                        task_num=32, update_step=5, update_step_test=5):

        super(Meta, self).__init__()

        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.task_num = task_num
        self.update_step = update_step
        self.update_step_test = update_step_test
        self.tasks = tasks
        self.gamma = gamma

        self.net = Learner(config)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def _sample_task_distribution(self):
        task = np.random.choice(self.tasks, p=self.task_weights)
        return task.X, task.y, task.x_control

    def forward(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, pretrained=False):
        task_num, setsz, c_ = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)] 
        corrects = [0 for _ in range(self.update_step + 1)]

        if not pretrained:
            for i in range(task_num):

                logits = self.net(x_spt[i], vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i]) \
                    + (self.gamma * fairness_reg(y_spt[i], F.softmax(logits,dim=1), c_spt[i]))
                grad = torch.autograd.grad(loss, self.net.parameters())
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

                with torch.no_grad():
                    logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i]) \
                        + (self.gamma * fairness_reg(y_qry[i], F.softmax(logits_q,dim=1),c_qry[i]))
                    losses_q[0] += loss_q

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[0] = corrects[0] + correct

                with torch.no_grad():
                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i]) \
                        + (self.gamma * fairness_reg(y_qry[i], F.softmax(logits_q,dim=1),c_qry[i]))
                    
                    losses_q[1] += loss_q
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[1] = corrects[1] + correct

                for k in range(1, self.update_step):
                    logits = self.net(x_spt[i], fast_weights, bn_training=True)

                    loss = F.cross_entropy(logits, y_spt[i]) 
                    add = (self.gamma * fairness_reg(y_spt[i], F.softmax(logits,dim=1),c_spt[i]))
                    loss += add
                    grad = torch.autograd.grad(add, fast_weights)

                    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i]) \
                        + (self.gamma * fairness_reg(y_qry[i], F.softmax(logits_q,dim=1),c_qry[i]))
                    losses_q[k + 1] += loss_q

                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct = torch.eq(pred_q, y_qry[i]).sum().item() 
                        corrects[k + 1] = corrects[k + 1] + correct

            loss_q = losses_q[-1] / task_num

            self.meta_optim.zero_grad()
            loss_q.backward()
            self.meta_optim.step()

            accs = np.array(corrects) / (querysz * task_num)
            return accs

        else:
            for i in range(task_num):
                logits = self.net(x_spt[i], vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i]) \
                    + (self.gamma * fairness_reg(y_spt[i], F.softmax(logits,dim=1), c_spt[i]))

                self.meta_optim.zero_grad()
                loss.backward()

                self.meta_optim.step()

    def finetunning(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry):
        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]
        dis = [0 for _ in range(self.update_step_test + 1)]

        net = deepcopy(self.net)

        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt) 
        
        add = (self.gamma * fairness_reg(y_spt, F.softmax(logits,dim=1),c_spt))
        
        loss += add
        grad = torch.autograd.grad(loss, net.parameters())

        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))


        with torch.no_grad():
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct
            dis[0] = fairness_notion(y_qry.data.numpy(), pred_q.data.numpy(), c_qry.data.numpy())

        with torch.no_grad():
            logits_q = net(x_qry, fast_weights, bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            
            correct = torch.eq(pred_q, y_qry).sum().item()
            
            corrects[1] = corrects[1] + correct
            dis[1] = fairness_notion(y_qry.data.numpy(), pred_q.data.numpy(), c_qry.data.numpy())

        for k in range(1, self.update_step_test):
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt) 
            add = (self.gamma * fairness_reg(y_spt, F.softmax(logits,dim=1),c_spt))
            loss += add
            grad = torch.autograd.grad(loss, fast_weights)

            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct
                dis[k + 1] = fairness_notion(y_qry.data.numpy(), pred_q.data.numpy(), c_qry.data.numpy())

        accs = np.array(corrects) / querysz
        dis = np.array(dis)

        return accs, dis, [net, fast_weights], deepcopy(self.net.parameters())

def main():
    config = get_MAML_config_cc()
    device = torch.device("cpu")

    num = 0

    for g in GAMMA_RANGE:
        print ("Beginning GAMMA {}".format(g))  
        print ("---------")


        # Setup meta model
        maml = Meta(config, device, gamma=g, update_step=2, update_step_test=5, update_lr=4e-1, meta_lr=1e-3)

        for step in range(EPOCHS+1):
            x_spt, y_spt, x_qry, y_qry, c_spt, c_qry = gen_batch(False)

            x_spt, x_qry, y_spt, y_qry = totorch(x_spt, device), totorch(x_qry, device), torch.from_numpy(y_spt), torch.from_numpy(y_qry)
            c_spt, c_qry = torch.from_numpy(c_spt), torch.from_numpy(c_qry)
            accs = maml(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry)

            # Display fine-tuning results every N epochs
            if step % 100 == 0:
                accs, dis = [], []
                for _ in range(NUM_TESTING_TASKS):
                    x_spt, y_spt, x_qry, y_qry, c_spt, c_qry = gen_batch(True)

                    x_spt = torch.from_numpy(x_spt).float()
                    y_spt = torch.from_numpy(y_spt).long() 
                    c_spt = torch.from_numpy(c_spt).long()

                    cur_indc = 0

                    updated_weights = []

                    for x_spt_one, y_spt_one, x_qry_one, y_qry_one, c_spt_one, c_qry_one in zip(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry):
                        x_qry_one = torch.from_numpy(x_qry_one).float()
                        y_qry_one = torch.from_numpy(y_qry_one).long()
                        c_qry_one = torch.from_numpy(c_qry_one).long()

                        test_acc, test_di, net, net_before = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one, c_spt_one, c_qry_one)

                        
                        updated_weights.append(net[1])

                        accs.append(test_acc)
                        dis.append(test_di)

                        del net

                        cur_indc += 1 

                cis_acc = np.std(accs,axis=0)
                cis_dis = np.std(dis,axis=0)

                accs = np.array(accs).mean(axis=0).astype(np.float16)
                dis = np.array(dis).mean(axis=0).astype(np.float16)

                print ("Result after epoch {}".format(step))
                print ("---------")
                print ("Avg tuning Acc:",accs)
                print ("Avg tuning {}:".format(METRIC),dis)
                print ("---------")

if __name__ == '__main__':
    main()
