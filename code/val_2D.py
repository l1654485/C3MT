import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


# def calculate_metric_percase(pred, gt):
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     if pred.sum() > 0:
#         dice = metric.binary.dc(pred, gt)
#         hd95 = metric.binary.hd95(pred, gt)
#         return dice, hd95
#     else:
#         return 0, 0

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

def calculate_metric_percase3(pred, gt):
    pred = pred.astype(int)
    gt = gt.astype(int)
    if pred.sum() > 0:
        Precison = metric.binary.precision(pred, gt)#预测为正类的样本中有多少是真正正确的
        Sensitivity = metric.binary.sensitivity(pred, gt)#所有正例中被分对的比例
        Specificity = metric.binary.specificity(pred, gt)#所有负例中被分对的比例
        return [Precison, Sensitivity, Specificity]
    else:
        return [0, 0, 0]

def test_single_volume1(image, label, net, classes, patch_size=[256, 256]):
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
            out, _, _ = net(input)  # 获取模型输出
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
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

def test_single_volume3(image, label, net, classes, patch_size=[256, 256]):
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
        metric_list.append(calculate_metric_percase3(prediction == i, label == i))
    return metric_list

def normalize(tensor):
    min_val = tensor.min(1, keepdim=True)[0]
    max_val = tensor.max(1, keepdim=True)[0]
    result = tensor - min_val
    result = result / max_val
    return result
def test_single_volume4(image, label, net1, net2,classes, patch_size=[224, 224]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()  # 降维
    prediction = np.zeros_like(label)  # 创建值为0的镜像标签
    for ind in range(image.shape[0]):  # 对体积中的每层切片分割
        slice = image[ind, :, :]  # 读取切片
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)  # 缩放切片
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()  # 将切片载入GPU
        net1.eval()  # 评估模式
        net2.eval()  # 评估模式
        with torch.no_grad():  # 禁止反向传播
            outputs_weak_soft1 = torch.softmax(net1(input), dim=1)
            pseudo_mask1 = (normalize(outputs_weak_soft1) > 0.95).float()  # 阈值选择
            outputs_weak_soft2 = torch.softmax(net2(input), dim=1)
            pseudo_mask2 = (normalize(outputs_weak_soft2) > 0.95).float()  # 阈值选择
            outputs_weak_soft_masked1 = (normalize(outputs_weak_soft1)) * pseudo_mask1
            outputs_weak_soft_masked2 = (normalize(outputs_weak_soft2)) * pseudo_mask2
            outputs_weak_soft_masked = (outputs_weak_soft_masked1 + outputs_weak_soft_masked2) / 2
            pseudo_outputs = torch.argmax(outputs_weak_soft_masked, dim=1, keepdim=False).squeeze(0)
            out = pseudo_outputs.cpu().detach().numpy()  # 将输出读入CPU
            # print(out.sum())
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)  # 缩放输出
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):  # 对每个类评估
        metric_list.append(calculate_metric_percase3(prediction == i, label == i))
    return metric_list

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



def show_single_volume1(image, label, net, classes, patch_size=[256, 256]):
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
            out, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                out, dim=1), dim=1).squeeze(0)  # 获取模型输出
            out = out.cpu().detach().numpy()  # 将输出读入CPU
            # print(out.sum())
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)  # 缩放输出
            prediction[ind] = pred
    return prediction

def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_volume_Vnet(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()  # 降维
    prediction = np.zeros_like(label)  # 创建值为0的镜像标签
    for ind in range(image.shape[0]):  # 对体积中的每层切片分割
        slice = image[ind, :, :]  # 读取切片
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)  # 缩放切片
        input = torch.from_numpy(slice).unsqueeze(
            2).unsqueeze(0).float().cuda()  # 将切片载入GPU
        net.eval()  # 评估模式
        with torch.no_grad():  # 禁止反向传播
            out = net(input)
            out = extract_prediction(out)
            out = out.squeeze(-1)
            out = out.unsqueeze(0)
            out = torch.softmax(out, dim=1)
            out = torch.argmax(out, dim=1) # 获取模型输出
            out = out.squeeze(0)
            out = out.cpu().detach().numpy()  # 将输出读入CPU
            # print(out.sum())
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)  # 缩放输出
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):  # 对每个类评估
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list

def test_single_volume_Vnet2(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()  # 降维
    prediction = np.zeros_like(label)  # 创建值为0的镜像标签
    for ind in range(image.shape[0]):  # 对体积中的每层切片分割
        slice = image[ind, :, :]  # 读取切片
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)  # 缩放切片
        input = torch.from_numpy(slice).unsqueeze(
            2).unsqueeze(0).unsqueeze(0).float().cuda()  # 将切片载入GPU
        net.eval()  # 评估模式
        with torch.no_grad():  # 禁止反向传播
            out = net(input)
            out = extract_prediction(out)
            out = out.squeeze(-1)
            out = torch.softmax(out, dim=1)
            out = torch.argmax(out, dim=1) # 获取模型输出
            out = out.squeeze(0)
            out = out.cpu().detach().numpy()  # 将输出读入CPU
            # print(out.sum())
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)  # 缩放输出
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):  # 对每个类评估
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list

def extract_prediction(output):
    """
    健壮地从模型输出中提取预测张量
    支持格式:
      - 张量 (直接返回)
      - 元组 (取第一个元素)
      - 字典 (尝试常见键名)
    """
    # 如果已经是张量，直接返回
    if isinstance(output, torch.Tensor):
        return output

    # 处理元组/列表
    if isinstance(output, (tuple, list)):
        return output[0]

    # 处理字典
    if isinstance(output, dict):
        return output["pred"]
def test_single_volume_test(image, label, net, classes, patch_size=[256, 256]):
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
            out = net(input)
            out = extract_prediction(out)
            out = torch.softmax(out, dim=1)
            out = torch.argmax(out, dim=1)  # 获取模型输出
            out = out.squeeze(0)
            out = out.cpu().detach().numpy()  # 将输出读入CPU
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)  # 缩放输出
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):  # 对每个类评估
        metric_list.append(calculate_metric_percase2(prediction == i, label == i))
    return metric_list

def test_single_volume_test1(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()  # 降维
    prediction = np.zeros_like(label)  # 创建值为0的镜像标签
    metric_list = []
    for ind in range(image.shape[0]):  # 对体积中的每层切片分割
        slice = image[ind, :, :]  # 读取切片
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)  # 缩放切片
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()  # 将切片载入GPU
        net.eval()  # 评估模式
        with torch.no_grad():  # 禁止反向传播
            out = net(input)
            out = extract_prediction(out)
            out = torch.softmax(out, dim=1)
            out = torch.argmax(out, dim=1)  # 获取模型输出
            out = out.squeeze(0)
            out = out.cpu().detach().numpy()  # 将输出读入CPU
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)  # 缩放输出
            for i in range(1, classes):  # 对每个类评估
                dice = calculate_metric_percase(pred == i, label[ind] == i)[0]
                metric_list.append(dice)
    return metric_list

def show_single_volume(image, label, net, classes, patch_size=[256, 256]):
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
            out = net(input)
            out = extract_prediction(out)
            out = torch.softmax(out, dim=1)
            out = torch.argmax(out, dim=1)  # 获取模型输出
            out = out.squeeze(0)
            out = out.cpu().detach().numpy()  # 将输出读入CPU
            # print(out.sum())
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)  # 缩放输出
            prediction[ind] = pred
    return prediction
