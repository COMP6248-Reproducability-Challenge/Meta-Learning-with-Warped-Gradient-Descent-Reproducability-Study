"""Meta-learners for Omniglot experiment.
Based on original implementation:
https://github.com/amzn/metalearn-leap
"""
import random
from abc import abstractmethod
from torch import nn as nn
from torch import optim
import torch

import optim
import updaters
import warpgrad

from utilsOmni import Res, AggRes

import time
import copy


class BaseWrapper(object):

    """Generic training wrapper.

    Arguments:
        criterion (func): loss criterion to use.
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
    """

    def __init__(self, criterion, model, optimizer_cls, optimizer_kwargs, args):
        self.criterion = criterion
        self.model = model
        self.args = args
        self.optimizer_cls = \
            optim.SGD if optimizer_cls.lower() == 'sgd' else optim.Adam
        self.optimizer_kwargs = optimizer_kwargs

    def __call__(self, tasks, meta_train=True):
        return self.run_tasks(tasks, meta_train=meta_train)

    @abstractmethod
    def _partial_meta_update(self, loss, final):
        """Meta-model specific meta update rule.

        Arguments:
            loss (nn.Tensor): loss value for given mini-batch.
            final (bool): whether iteration is the final training step.
        """
        NotImplementedError('Implement in meta-learner class wrapper.')

    @abstractmethod
    def _final_meta_update(self):
        """Meta-model specific meta update rule."""
        NotImplementedError('Implement in meta-learner class wrapper.')

    def run_tasks(self, tasks, meta_train):
        """Train on a mini-batch tasks and evaluate test performance.

        Arguments:
            tasks (list, torch.utils.data.DataLoader): list of task-specific
                dataloaders.
            meta_train (bool): whether current run in during meta-training.
        """
        results = []

        t_train = 0
        t_val = 0

        for task in tasks:

            task.dataset.train()

            t_train_buff = time.time()
            trainres = self.run_task(task, train=True, meta_train=meta_train)
            t_train += time.time() - t_train_buff

            task.dataset.eval()

            t_val_buf = time.time()
            valres = self.run_task(task, train=False, meta_train=False)
            t_val += time.time() - t_val_buf

            results.append((trainres, valres))

        for i in range(len(tasks)):
            tasks[i].dataset.train()
            trainres1 = self.run_task(tasks[i], train=True, meta_train=meta_train)


            tasks[i].dataset.eval()

            t_val_buf = time.time()
            valres = self.run_task(task, train=False, meta_train=False)
            t_val += time.time() - t_val_buf

            results.append((trainres, valres))

        ##
        results = AggRes(results)

        # Meta gradient step
        t_final_update = time.time()
        if meta_train:
            self._final_meta_update()
        t_final_update = time.time() - t_final_update

        #print("train", t_train)
        #print("val", t_val)
        #print("update", t_final_update)

        return results

    '''
    def run_tasks(self, tasks, meta_train):
        """Train on a mini-batch tasks and evaluate test performance.

        Arguments:
            tasks (list, torch.utils.data.DataLoader): list of task-specific
                dataloaders.
            meta_train (bool): whether current run in during meta-training.
        """
        results = []

        t_train = 0
        t_val = 0

        if torch.cuda.device_count() > 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            #model = MyDataParallel(model)

        dev1 = torch.device("cuda:0")
        dev2 = torch.device("cuda:1")

        model2 = copy.deepcopy(get_model(self.args, self.criterion))
        self.model.to(dev1)
        model2.to(dev2)

        for i in range(0, len(tasks), 2):
            task = tasks[i]
            task.dataset.train()

            task2 = tasks[i+1]
            task2.dataset.train()

            optimizer = None
            self.model.init_adaptation()
            self.model.train()
            optimizer = self.optimizer_cls(
                self.model.parameters(), **self.optimizer_kwargs)

            optimizer2 = None
            model2.init_adaptation()
            model2.train()
            optimizer2 = self.optimizer_cls(
                model2.parameters(), **self.optimizer_kwargs)

            res = Res()
            N = len(task)
            # batches = enumerate(batches).to(device, non_blocking=True)

            t1 = time.time()
            t_pred = 0
            t_back = 0

            it1 = iter(task)
            it2 = iter(task2)

            (input, target) = next(it1)
            (input2, target2) = next(it2)

            for n, (input, target) in enumerate(batches):
                input = input.to(dev1, non_blocking=True)
                target = target.to(dev1, non_blocking=True)

                input2 = input2.to(dev2, non_blocking=True)
                target2 = target2.to(dev2, non_blocking=True)

                # Evaluate model
                t_pred_bef = time.time()
                prediction = self.model(input)
                prediction2 = model2(input2)
                t_pred += (time.time() - t_pred_bef)
                # print(t_pred)
                loss = self.criterion(prediction, target)
                loss2 = self.criterion(prediction2, target2)

                # print("Outside: input size", input.size(),
                #      "output_size", prediction.size())

                res.log(loss=loss.item(), pred=prediction, target=target)
                res.log(loss=loss.item(), pred=prediction2, target=target2)

                # TRAINING #
                if not train:
                    continue

                final = (n + 1) == N
                t_back_bef = time.time()
                loss.backward()
                loss2.backward()
                t_back += (time.time() - t_pred_bef)
                # print(t_back)

                if meta_train:
                    self._partial_meta_update(loss, final)

                optimizer.step()
                optimizer.zero_grad()

                optimizer2.step()
                optimizer2.zero_grad()

                if final:
                    break
            ###
            t_agg = time.time()
            res.aggregate()
            # print("pred", t_pred)
            # print("back prop", t_back)
            # print("agg", time.time() - t_agg)
            return res


            tasks[i].dataset.eval()

            t_val_buf = time.time()
            valres = self.run_task(task, train=False, meta_train=False)
            t_val += time.time() - t_val_buf

            results.append((trainres, valres))

        ##
        results = AggRes(results)

        # Meta gradient step
        t_final_update = time.time()
        if meta_train:
            self._final_meta_update()
        t_final_update = time.time() - t_final_update

        #print("train", t_train)
        #print("val", t_val)
        #print("update", t_final_update)

        return results
        '''

    def run_task(self, task, train, meta_train):
        """Run model on a given task.

        Arguments:
            task (torch.utils.data.DataLoader): task-specific dataloaders.
            train (bool): whether to train on task.
            meta_train (bool): whether to meta-train on task.
        """
        optimizer = None
        if train:
            self.model.init_adaptation()
            self.model.train()
            optimizer = self.optimizer_cls(
                self.model.parameters(), **self.optimizer_kwargs)
        else:
            self.model.eval()

        return self.run_batches(
            task, optimizer, train=train, meta_train=meta_train)

    def run_batches(self, batches, optimizer, train=False, meta_train=False):
        """Iterate over task-specific batches.

        Arguments:
            batches (torch.utils.data.DataLoader): task-specific dataloaders.
            optimizer (torch.nn.optim): optimizer instance if training is True.
            train (bool): whether to train on task.
            meta_train (bool): whether to meta-train on task.
        """
        device = next(self.model.parameters()).device

        res = Res()
        N = len(batches)
        #batches = enumerate(batches).to(device, non_blocking=True)

        t1 = time.time()
        t_pred = 0
        t_back = 0

        for n, (input, target) in enumerate(batches):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Evaluate model
            t_pred_bef = time.time()
            prediction = self.model(input)
            t_pred += (time.time() - t_pred_bef)
            #print(t_pred)
            loss = self.criterion(prediction, target)

            #print("Outside: input size", input.size(),
            #      "output_size", prediction.size())

            res.log(loss=loss.item(), pred=prediction, target=target)

            # TRAINING #
            if not train:
                continue

            final = (n+1) == N
            t_back_bef = time.time()
            loss.backward()
            t_back += (time.time() - t_pred_bef)
            #print(t_back)

            if meta_train:
                self._partial_meta_update(loss, final)

            optimizer.step()
            optimizer.zero_grad()

            if final:
                break
        ###
        t_agg = time.time()
        res.aggregate()
        #print("pred", t_pred)
        #print("back prop", t_back)
        #print("agg", time.time() - t_agg)
        return res


class WarpGradWrapper(BaseWrapper):

    """Wrapper around WarpGrad meta-learners.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        meta_optimizer_cls: meta optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        meta_optimizer_kwargs (dict): kwargs to pass to meta optimizer upon
            construction.
        meta_kwargs (dict): kwargs to pass to meta-learner upon construction.
        criterion (func): loss criterion to use.
    """

    def __init__(self,
                 model,
                 optimizer_cls,
                 meta_optimizer_cls,
                 optimizer_kwargs,
                 meta_optimizer_kwargs,
                 meta_kwargs,
                 criterion, args):

        replay_buffer = warpgrad.ReplayBuffer(
            inmem=meta_kwargs.pop('inmem', True),
            tmpdir=meta_kwargs.pop('tmpdir', None))

        optimizer_parameters = warpgrad.OptimizerParameters(
            trainable=meta_kwargs.pop('learn_opt', False),
            default_lr=optimizer_kwargs['lr'],
            default_momentum=optimizer_kwargs['momentum']
            if 'momentum' in optimizer_kwargs else 0.)

        updater = updaters.DualUpdater(criterion, **meta_kwargs)

        model = warpgrad.Warp(model=model,
                              adapt_modules=list(model.adapt_modules()),
                              warp_modules=list(model.warp_modules()),
                              updater=updater,
                              buffer=replay_buffer,
                              optimizer_parameters=optimizer_parameters)

        super(WarpGradWrapper, self).__init__(criterion,
                                              model,
                                              optimizer_cls,
                                              optimizer_kwargs, args)

        self.meta_optimizer_cls = optim.SGD \
            if meta_optimizer_cls.lower() == 'sgd' else optim.Adam
        lra = meta_optimizer_kwargs.pop(
            'lr_adapt', meta_optimizer_kwargs['lr'])
        lri = meta_optimizer_kwargs.pop(
            'lr_init', meta_optimizer_kwargs['lr'])
        lrl = meta_optimizer_kwargs.pop(
            'lr_lr', meta_optimizer_kwargs['lr'])
        self.meta_optimizer = self.meta_optimizer_cls(
            [{'params': self.model.init_parameters(), 'lr': lri},
             {'params': self.model.warp_parameters(), 'lr': lra},
             {'params': self.model.optimizer_parameters(), 'lr': lrl}],
            **meta_optimizer_kwargs)

    def _partial_meta_update(self, loss, final):
        pass

    def _final_meta_update(self):

        def step_fn():
            self.meta_optimizer.step()
            self.meta_optimizer.zero_grad()

        self.model.backward(step_fn, **self.optimizer_kwargs)

    def run_task(self, task, train, meta_train):
        """Run model on a given task, first adapting and then evaluating"""
        if meta_train and train:
            # Register new task in buffer.
            self.model.register_task(task)
            self.model.collect()
        else:
            # Make sure we're not collecting non-meta-train data
            self.model.no_collect()

        optimizer = None
        if train:
            # Initialize model adaptation
            self.model.init_adaptation()

            optimizer = self.optimizer_cls(
                self.model.optimizer_parameter_groups(),
                **self.optimizer_kwargs)

            if self.model.collecting and self.model.learn_optimizer:
                # Register optimiser to collect potential momentum buffers
                self.model.register_optimizer(optimizer)
        else:
            self.model.eval()

        return self.run_batches(
            task, optimizer, train=train, meta_train=meta_train)

'''
class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
'''