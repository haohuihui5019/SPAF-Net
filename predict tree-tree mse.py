import argparse
import time
from torch.utils.data import DataLoader
import torch
from model.model_sup import vgg19 # if you wang to test fully-supervised model
# from model.model_semi import vgg19 # if you want to test semi-supervised model
from datasets.dataset import Crowd
import numpy as np
import os
def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--downsample-ratio', default=8, type=int,
                        help='the downsample ratio of the model')
    parser.add_argument('--data-dir', default='resize_class\\test',
                        help='the directory of the data')
    parser.add_argument('--model-path', default=r'history/sha_100_best_model.pth', 
                        help='the path to the model')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='the number of samples in a batch')
    parser.add_argument('--device', default='0',
                        help="assign device")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arg()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    dataset = Crowd(args.data_dir, 224, args.downsample_ratio, method='val')
    dataloader = DataLoader(dataset, 1, shuffle=False, pin_memory=False)
    model = vgg19(25)
    device = torch.device('cuda')
    model.to(device)
    checkpoint = torch.load(args.model_path, device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    file = open('result.txt', 'w')
    step = 0
    epoch_res = []
    epoch_res_bud = []
    epoch_res_bloom = []
    epoch_res_faded = []
    epoch_start = time.time()
    count_of_tree = dict()
    list = []
    for inputs, gt_counts, name in dataloader:
        if name[0][0:1] == "D":
            tree_name = name[0][0:-2]
            if tree_name.endswith("_"):
                tree_name = tree_name[:-1]
        else:
            tree_name = name[0].split('_')[0]

        if tree_name not in list:
            list.append(tree_name)
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)  # if you want to test fully-supervised model
            # outputs,_ = model(inputs) # if you want to test semi-supervised model

            pred_bud = outputs[: ,0:1, :, :]
            pred_bloom = outputs[: ,1:2, :, :]
            pred_faded = outputs[: ,2:3, :, :]

            res = sum(gt_counts) - torch.sum(outputs).item()
            res_bud = gt_counts[0] - torch.sum(pred_bud).item()
            res_bloom = gt_counts[1] - torch.sum(pred_bloom).item()
            res_faded = gt_counts[2] - torch.sum(pred_faded).item()

            if (tree_name + "_bud") not in count_of_tree:
                count_of_tree[tree_name + "_bud"] = [torch.sum(pred_bud).item(), float(str(gt_counts[0])[8:9])]
                count_of_tree[tree_name + "_bloom"] = [torch.sum(pred_bloom).item(), float(str(gt_counts[1])[8:9])]
                count_of_tree[tree_name + "_faded"] = [torch.sum(pred_faded).item(), float(str(gt_counts[2])[8:9])]
            else:
                count_of_tree[tree_name + "_bud"] = [torch.sum(pred_bud).item() + count_of_tree[tree_name + "_bud"][0],
                                                     float(str(gt_counts[0])[8:9]) + count_of_tree[tree_name + "_bud"][
                                                         1]]
                count_of_tree[tree_name + "_bloom"] = [
                    torch.sum(pred_bloom).item() + count_of_tree[tree_name + "_bloom"][0],
                    float(str(gt_counts[1])[8:9]) + count_of_tree[tree_name + "_bloom"][1]]
                count_of_tree[tree_name + "_faded"] = [
                    torch.sum(pred_faded).item() + count_of_tree[tree_name + "_faded"][0],
                    float(str(gt_counts[2])[8:9]) + count_of_tree[tree_name + "_faded"][1]]

    for tree in list:

        bud_mae = abs(count_of_tree[tree + "_bud"][0] - count_of_tree[tree + "_bud"][1])
        bloom_mae = abs(count_of_tree[tree + "_bloom"][0] - count_of_tree[tree + "_bloom"][1])
        faded_mae = abs(count_of_tree[tree + "_faded"][0] - count_of_tree[tree + "_faded"][1])
        epoch_res_bud.append(bud_mae)
        epoch_res_bloom.append(bloom_mae)
        epoch_res_faded.append(faded_mae)

    mse_bud = np.sqrt(np.mean(np.square(epoch_res_bud)))
    mae_bud = np.mean(np.abs(epoch_res_bud))
    mse_anthesis = np.sqrt(np.mean(np.square(epoch_res_bloom)))
    mae_anthesis = np.mean(np.abs(epoch_res_bloom))
    mse_petal_fall = np.sqrt(np.mean(np.square(epoch_res_faded)))
    mae_petal_fall = np.mean(np.abs(epoch_res_faded))

    print(
        'MAE_bud: {:.2f},MAE_anthesis: {:.2f}, MAE_petal-fall: {:.2f}, MSE_bud: {:.2f}, MSE_anthesis: {:.2f}, MSE_petal_fall: {:.2f} Cost {:.1f} sec'.format(
            mae_bud, mae_anthesis , mae_petal_fall, mse_bud, mse_anthesis, mse_petal_fall, (time.time() - epoch_start)))
