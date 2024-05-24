# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
import gc
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch
from torch.nn import functional as F
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.evaluation import SemSegEvaluator
from detectron2.utils.visualizer import ColorMode, Visualizer

# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.utils.colormap import random_color

from detectron2.evaluation import SemSegEvaluator

from baselines.utils.visualizer import CustomVisualizer
from baselines.utils.vis_uitls import show_image_relevance


# TODO:
json_index_counter = 0


class PartCLIPSegEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
        post_process_func=None,
        ORACLE=False,
        visualize=True,
        visualize_attn=False
    ):
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )
        meta = MetadataCatalog.get(dataset_name)
        try:
            self._evaluation_set = meta.evaluation_set
        except AttributeError:
            self._evaluation_set = None
        self.post_process_func = (
            post_process_func
            if post_process_func is not None
            else lambda x, **kwargs: x
        )
        self.oracle = ORACLE
        self.visualize = visualize
        self.visualize_attn = visualize_attn

        if self.visualize:
            self.vis_path = os.path.join(self._output_dir, "visualization")
            PathManager.mkdirs(self.vis_path)

        self.meta = meta

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        with torch.no_grad():
            for input, output in zip(inputs, outputs):
                if self.visualize_attn:
                    attns = output["attns"]
                obj_mask = input["instances"].gt_masks[0]

                sem_seg_output, sem_seg_output_all = output["sem_seg"], output["sem_seg_with_bg"]
                obj_sem_seg_output_all = output["obj_sem_seg_with_bg"]
                part_sem_seg_output = output["part_sem_seg"]

                output = self.post_process_func(sem_seg_output, image=np.array(Image.open(input["file_name"])))
                output_all = self.post_process_func(sem_seg_output_all, image=np.array(Image.open(input["file_name"])))
                obj_output_all = self.post_process_func(obj_sem_seg_output_all, image=np.array(Image.open(input["file_name"])))
                part_output = self.post_process_func(part_sem_seg_output, image=np.array(Image.open(input["file_name"])))

                output = output.argmax(dim=0).to(self._cpu_device)
                output_all = output_all.argmax(dim=0).to(self._cpu_device)
                obj_output_all = obj_output_all.to(self._cpu_device)
                part_output = part_output.argmax(dim=0).to(self._cpu_device)

                obj_mask = F.interpolate(obj_mask.float().unsqueeze(0).unsqueeze(0), size=output.shape[-2:], mode='nearest').squeeze()
                output[obj_mask == 0.0] = self.meta.ignore_label

                pred = np.array(output, dtype=np.int)
                pred_all = np.array(output_all, dtype=np.int)
                obj_pred_all = np.array(obj_output_all, dtype=np.int)
                part_pred = np.array(part_output, dtype=np.int)

                gt_classes = input["obj_part_instances"].gt_classes
                gt_masks = input["obj_part_instances"].gt_masks
                if len(gt_masks) == 0:
                    gt = np.zeros_like(pred) + self._ignore_label
                else:
                    gt = np.zeros_like(gt_masks[0], dtype=np.float) + self._ignore_label
                    for i in range(len(gt_classes)):
                        gt[gt_masks[i] == True] = gt_classes[i]
                    eval_image_size = pred.shape[-2:]
                    gt = F.interpolate(
                        torch.tensor(gt).unsqueeze(0).unsqueeze(0),
                        size=eval_image_size,
                        mode='nearest'
                    ).squeeze()
                    gt = gt.int().numpy()
                pred[pred == self._ignore_label] = self._num_classes
                gt[gt == self._ignore_label] = self._num_classes
                self._conf_matrix += np.bincount(
                    (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1)
                    if self.oracle
                    else (self._num_classes + 1) * pred_all.reshape(-1) + gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)

                self._predictions.extend(self.encode_json_sem_seg(
                    pred if self.oracle else pred_all, input["file_name"]))

                if self.visualize:
                    ext = os.path.splitext(input["file_name"])[1]

                    input_img_tensor = F.interpolate(
                        input["image"].unsqueeze(0),
                        size=eval_image_size,
                        mode='bilinear'
                    ).squeeze()
                    input_img_npy = input_img_tensor.permute(1, 2, 0).int().numpy()
                    input_img_pil = Image.fromarray(input_img_npy.astype(np.uint8))

                    if self.visualize_attn:
                        for tag, attn in attns.items():
                            attn_tensor = attn.unsqueeze(0)

                            attn = attn.squeeze(0).cpu().numpy()
                            attn = (attn * 255.).astype(np.uint8)
                            attn_img = Image.fromarray(attn).convert('L')
                            attn_img = attn_img.resize((input["image"].shape[-1], input["image"].shape[-2]))
                            attn_img.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, f"_{tag}.jpg")))

                            attn_npy = show_image_relevance(attn_tensor, input_img_pil)
                            attn_img = Image.fromarray(attn_npy)
                            attn_img.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, f"_relevance_{tag}.jpg")))

                    obj_pred_all_pil = Image.fromarray(obj_pred_all.astype(np.uint8)).convert('P')
                    obj_pred_all_pil.putpalette([v for color in self.meta.obj_colors for v in color])
                    obj_pred_all_pil = obj_pred_all_pil.convert('RGB')
                    obj_pred_all_pil = Image.blend(input_img_pil, obj_pred_all_pil, 0.5)
                    obj_pred_all_pil.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, "_obj_all_raw.jpg")))

                    part_pred_pil = Image.fromarray(part_pred.astype(np.uint8)).convert('P')
                    part_pred_pil.putpalette([v for color in self.meta.part_colors for v in color])
                    part_pred_pil = part_pred_pil.convert('RGB')
                    part_pred_pil = Image.blend(input_img_pil, part_pred_pil, 0.5)
                    part_pred_pil.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, "_part_raw.jpg")))

                    visualizer_pred = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)
                    visualizer_pred_all = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)
                    visualizer_obj_pred_all = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)
                    visualizer_part_pred = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)
                    visualizer_gt = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)

                    vis_pred = visualizer_pred.draw_sem_seg(pred)
                    vis_pred.save(os.path.join(self.vis_path, os.path.basename(input["file_name"])))

                    vis_pred_all = visualizer_pred_all.draw_sem_seg(pred_all)
                    vis_pred_all.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, "_all.jpg")))

                    vis_obj_pred_all = visualizer_obj_pred_all.draw_obj_sem_seg(obj_pred_all)
                    vis_obj_pred_all.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, "_obj_all.jpg")))

                    vis_part_pred = visualizer_part_pred.draw_part_sem_seg(part_pred)
                    vis_part_pred.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, "_part.jpg")))

                    vis_gt = visualizer_gt.draw_sem_seg(gt)
                    vis_gt.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, "_gt.jpg")))

            torch.cuda.empty_cache()
            gc.collect()

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        # TODO: epoch
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            global json_index_counter
            json_index_counter += 1
            file_path = os.path.join(self._output_dir, f"sem_seg_predictions_{json_index_counter:04d}.json")
            # file_path = os.path.join(self._output_dir, f"sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=np.float)
        iou = np.full(self._num_classes, np.nan, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        # tp = self._conf_matrix.diagonal().astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        # pos_gt = np.sum(self._conf_matrix, axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
        # pos_pred = np.sum(self._conf_matrix, axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]
        if self._evaluation_set is not None:
            for set_name, set_inds in self._evaluation_set.items():
                iou_list = []
                set_inds = np.array(set_inds, np.int)
                mask = np.zeros((len(iou),)).astype(np.bool)
                mask[set_inds] = 1
                miou = np.sum(iou[mask][acc_valid[mask]]) / np.sum(iou_valid[mask])
                pacc = np.sum(tp[mask]) / np.sum(pos_gt[mask])
                res["mIoU-{}".format(set_name)] = 100 * miou
                res["pAcc-{}".format(set_name)] = 100 * pacc
                iou_list.append(miou)
                miou = np.sum(iou[~mask][acc_valid[~mask]]) / np.sum(iou_valid[~mask])
                pacc = np.sum(tp[~mask]) / np.sum(pos_gt[~mask])
                res["mIoU-un{}".format(set_name)] = 100 * miou
                res["pAcc-un{}".format(set_name)] = 100 * pacc
                iou_list.append(miou)
        res['h-IoU'] = 2 * (res['mIoU-base'] * res['mIoU-unbase']) / (res['mIoU-base'] + res['mIoU-unbase'])
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results
