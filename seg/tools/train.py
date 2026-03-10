# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Provide args as argument to main()
# - Snapshot source code
# - Build UDA model instead of regular one
# - Add deterministic config flag

import argparse
import copy
import os
import os.path as osp
import sys
import time

# Ensure local project package imports (mmseg) work even when PYTHONPATH is
# not exported correctly in batch scripts.
_PROJ_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models.builder import build_train_model
from mmseg.utils import collect_env, get_root_logger
from mmseg.utils.collect_env import gen_code_archive


def parse_args(args):
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--check-pretrained',
        action='store_true',
        help='Fail fast if configured pretrained checkpoint does not exist')
    args = parser.parse_args(args)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args




def _extract_pretrained_path(cfg):
    model_cfg = cfg.get('model', {})
    if isinstance(model_cfg, dict):
        if model_cfg.get('pretrained', None) is not None:
            return model_cfg.get('pretrained')
        inner = model_cfg.get('model', None)
        if isinstance(inner, dict) and inner.get('pretrained', None) is not None:
            return inner.get('pretrained')
    return None


def _log_pretrained_check(cfg, logger, strict=False):
    pretrained = _extract_pretrained_path(cfg)
    if not pretrained:
        logger.warning('Pretrained check: no `model.pretrained` path found in config.')
        if strict:
            raise RuntimeError('Pretrained check failed: no pretrained path in config.')
        return

    if osp.isabs(pretrained):
        resolved = pretrained
    else:
        resolved = osp.join(osp.dirname(__file__), '..', pretrained)
        resolved = osp.abspath(resolved)

    exists = osp.exists(resolved)
    logger.info(f'Pretrained check: config pretrained={pretrained}')
    logger.info(f'Pretrained check: resolved path={resolved}')
    logger.info(f'Pretrained check: exists={exists}')

    if strict and not exists:
        raise FileNotFoundError(
            f'Pretrained check failed: expected checkpoint not found at {resolved}')


def _log_weight_fingerprint(model, logger):
    base = model.module if hasattr(model, 'module') else model
    target = base
    if hasattr(base, 'model'):
        target = base.model

    backbone = getattr(target, 'backbone', None)
    if backbone is None:
        logger.warning('Pretrained check: backbone module not found for fingerprint logging.')
        return

    first_param = None
    for _, p in backbone.named_parameters():
        first_param = p
        break

    if first_param is None:
        logger.warning('Pretrained check: backbone has no parameters.')
        return

    mean_abs = float(first_param.detach().abs().mean().cpu().item())
    std = float(first_param.detach().std().cpu().item())
    logger.info(
        f'Pretrained check: backbone first-param stats mean_abs={mean_abs:.6e}, std={std:.6e}')

def main(args):
    args = parse_args(args)

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    cfg.model.train_cfg.work_dir = cfg.work_dir
    cfg.model.train_cfg.log_config = cfg.log_config
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # snapshot source code
    gen_code_archive(cfg.work_dir)
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is None and 'seed' in cfg:
        args.seed = cfg['seed']
    if args.seed is not None:
        deterministic = args.deterministic or cfg.get('deterministic')
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{deterministic}')
        set_random_seed(args.seed, deterministic=deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.splitext(osp.basename(args.config))[0]

    _log_pretrained_check(cfg, logger, strict=args.check_pretrained)

    model = build_train_model(
        cfg, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    _log_weight_fingerprint(model, logger)

    logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main(sys.argv[1:])
