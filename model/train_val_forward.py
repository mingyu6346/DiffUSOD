import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# from loss import structure_loss


def normalize_to_01(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def simple_train_val_forward(model: nn.Module, gt=None, image=None, **kwargs):
    if model.training:
        assert gt is not None and image is not None
        return model(gt, image, **kwargs)
    else:
        time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False
        gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
        pred = model.sample(image, **kwargs)
        if time_ensemble:
            preds = torch.concat(model.history, dim=1).detach().cpu()
            pred = torch.mean(preds, dim=1, keepdim=True)

            def process(i, p, gt_size):
                p = F.interpolate(p.unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                p = normalize_to_01(p)
                ps = F.interpolate(preds[i].unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                preds_round = (ps > 0).float().mean(dim=1, keepdim=True)
                p_postion = (preds_round > 0.5).float()
                p = p_postion * p
                return p

            pred = [process(index, p, gt_size) for index, (p, gt_size) in enumerate(zip(pred, gt_sizes))]
        return {
            "image": image,
            "pred": pred,
            "gt": gt if gt is not None else None,
        }


def calculate_entropy_pytorch(image_tensor):
    """
    使用PyTorch计算图像的信息熵
    :param image_tensor: 图像张量 (C, H, W 或 H, W)
    :return: 熵值
    """
    image_tensor = normalize_to_01(image_tensor)

    # 将图像展平
    image_flat = image_tensor.flatten()

    # 计算直方图并归一化
    histogram = torch.histc(image_flat, bins=2, min=0.0, max=1.0)
    prob = histogram / torch.sum(histogram)  # 概率分布

    # 计算熵，忽略概率为0的部分
    prob_nonzero = prob[prob > 0]
    entropy = -torch.sum(prob_nonzero * torch.log2(prob_nonzero))
    return entropy.item()


def entropy_weight_fusion_pytorch(ps, p):
    """
    基于熵权法的图像融合 (使用PyTorch)
    :param ps: 多张灰度图像的张量列表，每个张量大小为 (H, W)
    :return: 融合后的图像张量 (H, W)
    """
    # 计算每张图像的信息熵
    entropies = torch.tensor([calculate_entropy_pytorch(img) for img in ps])
    # weights = 1 / entropies  # 熵的倒数作为权重
    weights = entropies  # 熵的倒数作为权重
    weights /= weights.sum()  # 归一化权重

    # 融合图像
    fused_image = torch.zeros_like(ps[0], dtype=torch.float32)
    for weight, img in zip(weights, ps):
        img = normalize_to_01(img)
        fused_image += weight * img.float()

    # 映射到 0-255 的范围
    # fused_image = torch.clamp(fused_image, 0, 255).byte()
    fused_image = (normalize_to_01(fused_image + p) > 0.5).float()
    return fused_image


def weight_merge(images):
    # 加载图片
    weights = [0.05, 0.10, 0.2, 0.30, 0.35]  # 权重和为1
    weights = sorted(weights, reverse=True)
    # weights = [0.1, 0.15, 0.2, 0.25, 0.3].reverse()  # 权重和为1

    # 加权平均
    merged_image = sum(w * normalize_to_01(img) for w, img in zip(weights, images))

    return merged_image





def modification_train_val_forward(model: nn.Module, gt=None, image=None, seg=None, **kwargs):
    """This is for the modification task. When diffusion model add noise, will use seg instead of gt."""
    if model.training:
        assert gt is not None and image is not None and seg is not None
        return model(gt, image, seg=seg, **kwargs)

    else:
        time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False

        gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
        pred = model.sample(image, **kwargs).detach().cpu()
        if time_ensemble:
            """ Here is the function 3, Uncertainty based"""
            # print(len(model.history))
            preds = torch.concat(model.history, dim=1).detach().cpu()
            pred = torch.mean(preds, dim=1, keepdim=True)

            def process(i, p, gt_size):
                p = F.interpolate(p.unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                p = normalize_to_01(p)
                ps = F.interpolate(preds[i].unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                preds_round = (ps > 0).float().mean(dim=1, keepdim=True)
                p_postion = (preds_round > 0.5).float()
                p = p_postion * p
                return p

            # def process(i, p, gt_size):
            #     p = F.interpolate(p.unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
            #     p_norm = normalize_to_01(p)  # 1,1,H,W  mean (0, 1)
            #
            #     p_norm_large = (p_norm > 0.5).float()  # 1,1,H,W  mean (0, 1)
            #
            #     # preds B, T, H, W
            #     # ps    1, T, H, W  (-1, 1)
            #     ps = F.interpolate(preds[i].unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
            #     preds_round = (ps > 0).float().mean(dim=1, keepdim=True)
            #     p_postion = (preds_round > 0.5).float()
            #     p = (normalize_to_01(p_postion * p + p_norm_large) > 0.5).float()

                # ps_norm = torch.cat([ps_norm, p], dim=1)
                # ps_median_values = torch.median(ps_norm, dim=1, keepdim=True).values
                # ps_median = (ps_median_values > 0.5).float()
                # preds_round = (ps > 0).float().mean(dim=1, keepdim=True)
                # p_postion = (preds_round >= 0.5).float()
                # p = p_postion  * ps_median

                # ps_norm_unfold = ps.view(1, -1, ps.size(2), ps.size(3))
                # # 计算每个像素位置上的众数
                # ps_norm_unfold_mode = torch.mode(ps_norm_unfold, dim=1)[0]
                #
                # # 将众数结果重新组合成一个形状为(1, 1, H, W)的张量
                # p = ps_norm_unfold_mode.view(1, 1, ps.size(2), ps.size(3)) * p

                return p

            pred = [process(index, p, gt_size) for index, (p, gt_size) in enumerate(zip(pred, gt_sizes))]
        return {
            "image": image,
            "pred": pred,
            "gt": gt if gt is not None else None
        }


def modification_train_val_forward_e(model: nn.Module, gt=None, image=None, seg=None, **kwargs):
    """This is for the modification task. When diffusion model add noise, will use seg instead of gt."""
    if model.training:
        assert gt is not None and image is not None and seg is not None
        return model(gt, image, seg=seg, **kwargs)
    else:
        time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False
        gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
        pred = model.sample(image, **kwargs).detach().cpu()
        if time_ensemble:
            """ Here is extend function 4, with batch extend."""
            preds = torch.concat(model.history, dim=1).detach().cpu()
            for i in range(2):
                model.sample(image, **kwargs)
                preds = torch.cat([preds, torch.concat(model.history, dim=1).detach().cpu()], dim=1)
            pred = torch.mean(preds, dim=1, keepdim=True)

            def process(i, p, gt_size):
                p = F.interpolate(p.unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                p = normalize_to_01(p)
                ps = F.interpolate(preds[i].unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                preds_round = (ps > 0).float().mean(dim=1, keepdim=True)
                p_postion = (preds_round > 0.5).float()
                p = p_postion * p
                return p

            pred = [process(index, p, gt_size) for index, (p, gt_size) in enumerate(zip(pred, gt_sizes))]
        return {
            "image": image,
            "pred": pred,
            "gt": gt if gt is not None else None,
        }
