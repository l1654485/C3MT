import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

def calculate_metric_percase(pred, gt):
    pred = pred.astype(int)
    gt = gt.astype(int)
    # print(pred.shape, gt.shape)
    # print(np.unique(pred))
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd = 0
        return [dice, hd]
    else:
        return [0, 0]

def calculate_metric_percase2(pred, gt):
    pred = pred.astype(int)
    gt = gt.astype(int)
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt) #IOU
        hd = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        pre = metric.binary.precision(pred, gt)
        sen = metric.binary.sensitivity(pred, gt)
        spec = metric.binary.specificity(pred, gt)
        return [dice, jc, hd, asd, pre, sen, spec]
    else:
        return [0, 0, 0, 0, 0, 0, 0]

def normalize(tensor):
    min_val = tensor.min(1, keepdim=True)[0]
    max_val = tensor.max(1, keepdim=True)[0]
    result = tensor - min_val
    result = result / max_val
    return result

def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()  # 降维
    prediction = np.zeros_like(label)  # 创建值为0的镜像标签
    for ind in range(image.shape[0]):  # 对体积中的每层切片分割
        slice = image[ind, :, :]  # 读取切片
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)  # 缩放切片
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()  # 将切片载入GPU
        net.eval()  # 评估模式
        with torch.no_grad():  # 禁止反向传播
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)  # 获取模型输出
            out = out.cpu().detach().numpy()  # 将输出读入CPU
            # print(out.sum())
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)  # 缩放输出
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):  # 对每个类评估
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list

def test_single_volume2(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()  # 降维
    prediction = np.zeros_like(label)  # 创建值为0的镜像标签
    for ind in range(image.shape[0]):  # 对体积中的每层切片分割
        slice = image[ind, :, :]  # 读取切片
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)  # 缩放切片
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()  # 将切片载入GPU
        net.eval()  # 评估模式
        with torch.no_grad():  # 禁止反向传播
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)  # 获取模型输出
            out = out.cpu().detach().numpy()  # 将输出读入CPU
            # print(out.sum())
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)  # 缩放输出
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):  # 对每个类评估
        metric_list.append(calculate_metric_percase2(prediction, label))
    return metric_list