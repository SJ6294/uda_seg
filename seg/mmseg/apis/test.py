# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support debug_output_attention

import os
import os.path as osp
import tempfile

import mmcv
import numpy as np
import torch
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info


def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name




def _extract_img_metas(data):
    img_metas = data.get('img_metas', None)
    if img_metas is None:
        return []
    payload = img_metas
    if isinstance(payload, list) and len(payload) > 0:
        payload = payload[0]
    if hasattr(payload, 'data'):
        payload = payload.data[0]
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, (list, tuple)):
        if len(payload) == 0:
            return []
        if isinstance(payload[0], dict):
            return list(payload)
        if isinstance(payload[0], (list, tuple)) and len(payload[0]) > 0 and                 isinstance(payload[0][0], dict):
            return list(payload[0])
    return []


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    save_concat=False,
                    concat_out_dir=None):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    if efficient_test:
        mmcv.mkdir_or_exist('.efficient_test')

    if save_concat:
        if concat_out_dir is None:
            concat_out_dir = 'result'
        mmcv.mkdir_or_exist(concat_out_dir)

    dataset_index = 0

    for i, data in enumerate(data_loader):
        model_data = data
        # Validation dataloaders that are built with test_mode=False provide
        # train-style tensors instead of test-time list wrappers.
        # forward_test expects list inputs, so wrap here for compatibility.
        if not isinstance(data.get('img', None), list):
            model_data = data.copy()
            model_data['img'] = [data['img']]

            meta_list = _extract_img_metas(data)
            model_data['img_metas'] = [meta_list]

        infer_data = dict(
            img=model_data['img'], img_metas=model_data['img_metas'])
        with torch.no_grad():
            result = model(return_loss=False, **infer_data)

        if show or out_dir:
            img_tensor = data['img'][0] if isinstance(data['img'], list) else data['img']
            img_metas = _extract_img_metas(data)
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                if hasattr(model.module.decode_head,
                           'debug_output_attention') and \
                        model.module.decode_head.debug_output_attention:
                    # Attention debug output
                    mmcv.imwrite(result[0] * 255, out_file)
                else:
                    model.module.show_result(
                        img_show,
                        result,
                        palette=dataset.PALETTE,
                        show=show,
                        out_file=out_file,
                        opacity=opacity)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result, tmpdir='.efficient_test')
            results.append(result)

        batch_size = len(result)
        if save_concat:
            img_metas = _extract_img_metas(data)
            for b in range(batch_size):
                img_meta = img_metas[b]
                sample_idx = dataset_index + b
                ann_rel_path = None
                if hasattr(dataset, 'img_infos') and sample_idx < len(dataset.img_infos):
                    ann_rel_path = dataset.img_infos[sample_idx].get('ann', {}).get('seg_map')

                if ann_rel_path is None:
                    continue

                gt_path = osp.join(dataset.ann_dir, ann_rel_path)
                gt = mmcv.imread(gt_path, flag='unchanged')
                if gt.ndim == 3:
                    gt = gt[:, :, 0]
                gt = (gt >= 128).astype(np.uint8)

                pred = result[b]
                if isinstance(pred, str):
                    pred = np.load(pred)
                pred = pred.astype(np.uint8)

                ori_img = mmcv.imread(img_meta['filename'], flag='color')
                if ori_img is None:
                    continue
                h, w = gt.shape[:2]
                ori_img = mmcv.imresize(ori_img, (w, h))

                pred_vis = np.stack([pred * 255, pred * 255, pred * 255], axis=-1)
                gt_vis = np.stack([gt * 255, gt * 255, gt * 255], axis=-1)
                concat_img = np.concatenate([ori_img, pred_vis, gt_vis], axis=1)

                base = osp.splitext(osp.basename(img_meta['ori_filename']))[0]
                save_path = osp.join(concat_out_dir, f'{base}.png')
                mmcv.imwrite(concat_img, save_path)

        dataset_index += batch_size
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    if efficient_test:
        mmcv.mkdir_or_exist('.efficient_test')

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result, tmpdir='.efficient_test')
            results.append(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results
