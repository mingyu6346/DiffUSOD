import os, math
# import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
from thop import profile


# def backup_empty_files(input_dir='/media/omnisky/jixie/zmy/project/Diffusion/results/pth',
#                        output_dir='/media/omnisky/jixie/zmy/project/Diffusion/results_empty/pth'):
#     # 如果输出路径不存在，则创建它
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     # 遍历输入路径内的所有文件和文件夹
#     for root, dirs, files in os.walk(input_dir):
#         # 计算相对路径
#         relative_path = os.path.relpath(root, input_dir)
#
#         # 在输出路径创建相同的目录结构
#         output_root = os.path.join(output_dir, relative_path)
#         if not os.path.exists(output_root):
#             os.makedirs(output_root)
#
#         # 为每个文件创建空文件
#         for file in files:
#             input_file_path = os.path.join(root, file)
#             output_file_path = os.path.join(output_root, file)
#
#             # 创建空文件
#             open(output_file_path, 'w').close()

    # print(f"空文件备份已完成，输出路径为: {output_dir}")


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * init_lr
        lr = param_group['lr']
    return lr


def adjust_lr_2(optimizer, epoch_list, lr_list, epoch):
    if epoch < epoch_list[0]:
        c_lr = lr_list[0]
    elif epoch < sum(epoch_list[:-1]):
        c_lr = lr_list[1]
    else:
        c_lr = lr_list[2]
    for param_group in optimizer.param_groups:
        param_group['lr'] = c_lr
        lr = param_group['lr']
    return lr


def opt_save(opt):
    log_path = opt.save_path + "train_settings.log"
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    att = [i for i in opt.__dir__() if not i.startswith("_")]
    with open(log_path, "w") as f:
        for i in att:
            print("{}:{}".format(i, eval(f"opt.{i}")), file=f)


def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()

def fps_diff(diff_model, epoch_num, size, gpu=0):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    ls = []  # 每次计算得到的fps
    iterations = 300  # 重复计算的轮次
    device = f'cuda:{gpu}' if gpu >= 0 else 'cpu'
    diff_model.eval().to(device)
    random_input = torch.randn(1, 3, size, size).to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    input = random_input


    # for _ in range(50):
    #     _ = model(*input)

    for i in range(epoch_num + 1):
        # 测速
        times = torch.zeros(iterations)  # 存储每轮iteration的时间
        with torch.inference_mode():
            for iter in range(iterations):
                starter.record()
                _ = diff_model.sample(input,verbose=False)
                ender.record()

                # 同步GPU时间
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)  # 计算时间
                times[iter] = curr_time
                # print(curr_time)

        mean_time = times.mean().item()

        if i == 0:
            print("Initialization Inference time: {:.2f} ms, FPS: {:.2f} ".format(mean_time, 1000 / mean_time))
        if i != 0:
            ls.append(1000 / mean_time)
            print("{}/{} Inference time: {:.2f} ms, FPS: {:.2f} ".format(i, epoch_num, mean_time, 1000 / mean_time))
    print(f"平均fps为 {np.mean(ls):.2f}")
    print(f"最大fps为 {np.max(ls):.2f}")

def fps(model, epoch_num, size, gpu=0, count=2):
    # dummy_input = torch.randn(1, 3, size, size).cuda()
    # # flops, params = profile(model, (dummy_input, ))
    # flops, params = profile(model, (dummy_input, dummy_input))
    # print('GFLOPs: %.2f , params: %.2f M' % (flops / 1.0e9, params / 1.0e6))

    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    assert count in [1, 2, 3]

    ls = []  # 每次计算得到的fps
    iterations = 300  # 重复计算的轮次
    device = f'cuda:{gpu}' if gpu >= 0 else 'cpu'
    model.eval().to(device)
    random_input0 = torch.randn(1, 1, size, size).to(device)
    random_input = torch.randn(1, 3, size, size).to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    input = []

    if count == 1:
        input = [random_input]
    elif count == 2:
        input = [random_input0, random_input]
    elif count == 3:
        T = torch.randn(size=(1,)).to(device)
        noise = torch.randn(size=(1, 1, size, size)).to(device)
        input = [noise, T, random_input]



    for i in range(epoch_num + 1):
        # 测速
        times = torch.zeros(iterations)  # 存储每轮iteration的时间
        with torch.inference_mode():
            for iter in range(iterations):
                starter.record()
                _ = model(*input)
                ender.record()

                # 同步GPU时间
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)  # 计算时间
                times[iter] = curr_time
                # print(curr_time)

        mean_time = times.mean().item()

        if i == 0:
            print("Initialization Inference time: {:.2f} ms, FPS: {:.2f} ".format(mean_time, 1000 / mean_time))
        if i != 0:
            ls.append(1000 / mean_time)
            print("{}/{} Inference time: {:.2f} ms, FPS: {:.2f} ".format(i, epoch_num, mean_time, 1000 / mean_time))
    print(f"平均fps为 {np.mean(ls):.2f}")
    print(f"最大fps为 {np.max(ls):.2f}")


def flops(model, size=384, gpu=0, count=3, batchsize=1):
    assert count in [-1, 1, 2, 3], 'please input correct param number !'
    device = f'cuda:{gpu}' if gpu >= 0 else 'cpu'
    model.eval().to(device)

    dummy_input0 = torch.randn(1, 1, size, size).to(device)
    dummy_input = torch.randn(1, 3, size, size).to(device)
    if count == 3:
        T = torch.randn(size=(batchsize,)).to(device)
        noise = torch.randn(size=(batchsize, 1, size, size)).to(device)
        flops, params = profile(model, (noise, T, dummy_input))
    elif count == 2:
        flops, params = profile(model, (dummy_input0, dummy_input))
    elif count == 1:
        flops, params = profile(model, (dummy_input,))
    else:
        T = torch.randn(size=(batchsize,)).to(device)
        noise = torch.randn(size=(batchsize, 1, size, size)).to(device)
        x1 = torch.randn(batchsize, 64, size//4, size//4).to(device)
        x2 = torch.randn(batchsize, 128, size//8, size//8).to(device)
        x3 = torch.randn(batchsize, 256, size//16, size//16).to(device)
        x4 = torch.randn(batchsize, 512, size//32, size//32).to(device)

        flops, params = profile(model, (noise,T,[x1,x2,x3,x4],))


    print('GFLOPs: %.5f , params: %.2f M' % (flops / 1.0e9, params / 1.0e6))


def send_wechat(title, msg):
    token = 'a4db4c2545f64eb0b8739823432b8086'
    content = msg
    template = 'html'
    url = f"https://www.pushplus.plus/send?token={token}&title={title}&content={content}&template={template}"
    # print(url)
    r = requests.get(url=url)
    print(r.text)


def metrics_dict_to_float(validate_metrics_dict: dict):
    maxem = validate_metrics_dict['maxem']
    avgem = validate_metrics_dict['avgem']

    sm = validate_metrics_dict['sm']

    maxfm = validate_metrics_dict['maxfm']
    avgfm = validate_metrics_dict['avgfm']

    mae = validate_metrics_dict['mae']

    max_performance = round((1 - mae) + maxem + sm + maxfm, 3)
    avg_performance = round((1 - mae) + avgem + sm + avgfm, 3)
    # print(f"max_performance: {max_performance:.3f} / 4.000      avg_performance: {avg_performance:.3f} / 4.000")
    # print(validate_metrics_dict)
    return max_performance, avg_performance


def seed_torch(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def param(model):
    param_sum = 0
    for i in model.named_modules():
        if '.' not in i[0] and i[0] != '':
            layer = getattr(model, i[0])
            temp = sum(p.numel() for p in layer.parameters() if p.requires_grad) / 1e6
            param_sum += temp
            print(i[0], temp, "M")
    print(param_sum, 'M')


def load_param(pth_path, model, mode=None):
    if mode == "state_dict":
        pretraind_dict = torch.load(pth_path)["state_dict"]
    elif mode == "model":
        pretraind_dict = torch.load(pth_path)["model"]
        print(pretraind_dict.keys())
    elif mode == "state_dict_ema":
        pretraind_dict = torch.load(pth_path)["state_dict_ema"]
        print(pretraind_dict.keys())
    elif mode == "url":
        pretraind_dict = pth_path
    else:
        pretraind_dict = torch.load(pth_path)
        print(pretraind_dict.keys())

    model_dict = model.state_dict()

    # 只将pth中那些在model中的参数，提取出来
    state_dict = {k: v for k, v in pretraind_dict.items() if k in model_dict.keys()}

    # 将提取出来的参数更新到model_dict中，而model_dict有的，而state_dict没有的参数，不会被更新
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    len_pretrained = len(pretraind_dict.keys())
    len_model = len(model_dict.keys())
    len_match = len(state_dict.keys())

    print(f"{mode=} {len_match} / {len_model} / {len_pretrained}")

    # print(
    #     f"Founding {len_pretrained} pairs of param in {pth_path}, need {len_model} pairs, loaded {len_match} pairs.\n")

def copy_net(cfg):
    net_path = "model/net.py"
    os.makedirs(cfg.results_folder,exist_ok=True)
    net_dest_path = os.path.join(cfg.results_folder, "net.py")
    with open(net_dest_path, "w") as f1:
        with open(net_path, "r") as f2:
            f1.write(f2.read())

if __name__ == '__main__':
    # send_wechat(title='训练完成', msg='nihao')
    x = torch.ones(1, 1, 384, 384)
    y = torch.zeros(1, 1, 384, 384)
    # backup_empty_files()
    send_wechat('title','nihao')
