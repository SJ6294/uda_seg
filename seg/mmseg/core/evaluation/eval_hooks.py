# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import os.path as osp

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, efficient_test=False, result_dir='result', **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test
        self.result_dir = result_dir

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmseg.apis import single_gpu_test
        result_dir = osp.join(runner.work_dir, self.result_dir)

        results = single_gpu_test(
            runner.model,
            self.dataloader,
            show=False,
            efficient_test=self.efficient_test,
            save_concat=True,
            concat_out_dir=result_dir)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)

        val_loss = self._get_val_loss(runner)
        runner.log_buffer.output['val_loss'] = val_loss

        bg_iou = runner.log_buffer.output.get('IoU.background')
        obj_iou = runner.log_buffer.output.get('IoU.object')
        miou = runner.log_buffer.output.get('mIoU')
        mmcv.print_log(
            f"Validation summary | IoU.background: {bg_iou} | IoU.object: {obj_iou} | mIoU: {miou} | val_loss: {val_loss}",
            logger=runner.logger)

        if self.save_best:
            self._save_ckpt(runner, key_score)

    def _get_val_loss(self, runner):
        model = runner.model
        model.eval()
        losses = []
        with torch.no_grad():
            for data in self.dataloader:
                loss_dict = model(return_loss=True, **data)
                if isinstance(loss_dict, dict):
                    total_loss = 0
                    for val in loss_dict.values():
                        if isinstance(val, torch.Tensor):
                            total_loss += val.mean()
                        elif isinstance(val, (list, tuple)):
                            total_loss += sum(v.mean() for v in val if isinstance(v, torch.Tensor))
                    losses.append(float(total_loss.item()))
        if not losses:
            return float('nan')
        return float(np.mean(losses))


class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, efficient_test=False, result_dir='result', **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test
        self.result_dir = result_dir

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmseg.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            efficient_test=self.efficient_test)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
