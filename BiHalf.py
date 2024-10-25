import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from tqdm import tqdm

from utils import get_base_data, prepare_data


def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


# faster but more memory
def CalcTopMapWithPR(qB, queryL, rB, retrievalL, topk):
    num_query = queryL.shape[0]
    num_gallery = retrievalL.shape[0]
    topkmap = 0
    prec = np.zeros((num_query, num_gallery))
    recall = np.zeros((num_query, num_gallery))
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        all_sim_num = np.sum(gnd)

        prec_sum = np.cumsum(gnd)
        return_images = np.arange(1, num_gallery + 1)

        prec[iter, :] = prec_sum / return_images
        recall[iter, :] = prec_sum / all_sim_num

        assert recall[iter, -1] == 1.0
        assert all_sim_num == prec_sum[-1]

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    index = np.argwhere(recall[:, -1] == 1.0)
    index = index.squeeze()
    prec = prec[index]
    recall = recall[index]
    cum_prec = np.mean(prec, 0)
    cum_recall = np.mean(recall, 0)

    return topkmap, cum_prec, cum_recall


# https://github.com/chrisbyd/DeepHash-pytorch/blob/master/validate.py
def validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset):
    device = config["device"]
    # print("calculating test binary code......")
    tst_binary, tst_label = compute_result(test_loader, net, device=device)

    # print("calculating dataset binary code.......")
    trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

    if "pr_curve_path" not in config:
        mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), config["topK"])
    else:
        # need more memory
        mAP, cum_prec, cum_recall = CalcTopMapWithPR(tst_binary.numpy(), tst_label.numpy(),
                                                     trn_binary.numpy(), trn_label.numpy(),
                                                     config["topK"])
        index_range = num_dataset // 100
        index = [i * 100 - 1 for i in range(1, index_range + 1)]
        max_index = max(index)
        overflow = num_dataset - index_range * 100
        index = index + [max_index + i for i in range(1, overflow + 1)]
        c_prec = cum_prec[index]
        c_recall = cum_recall[index]

        pr_data = {
            "index": index,
            "P": c_prec.tolist(),
            "R": c_recall.tolist()
        }
        os.makedirs(os.path.dirname(config["pr_curve_path"]), exist_ok=True)
        with open(config["pr_curve_path"], 'w') as f:
            f.write(json.dumps(pr_data))
        print("pr curve save to ", config["pr_curve_path"])

    if mAP > Best_mAP:
        Best_mAP = mAP
        if "save_path" in config:
            save_path = os.path.join(config["save_path"], f'{config["dataset"]}_{bit}bits_{mAP}')
            os.makedirs(save_path, exist_ok=True)
            print("save in ", save_path)
            np.save(os.path.join(save_path, "tst_label.npy"), tst_label.numpy())
            np.save(os.path.join(save_path, "tst_binary.npy"), tst_binary.numpy())
            np.save(os.path.join(save_path, "trn_binary.npy"), trn_binary.numpy())
            np.save(os.path.join(save_path, "trn_label.npy"), trn_label.numpy())
            torch.save(net.state_dict(), os.path.join(save_path, "model.pt"))
    print(f"{config['info']} epoch:{epoch + 1} bit:{bit} dataset:{config['dataset']} MAP:{mAP} Best MAP: {Best_mAP}")
    print(config)
    return Best_mAP


# Deep Unsupervised Image Hashing by Maximizing Bit Entropy(AAAI2021)
# paper [Deep Unsupervised Image Hashing by Maximizing Bit Entropy](https://arxiv.org/pdf/2012.12334.pdf)
# code [Deep-Unsupervised-Image-Hashing](https://github.com/liyunqianggyn/Deep-Unsupervised-Image-Hashing)
# [BiHalf Unsupervised] epoch:40, bit:64, dataset:cifar10-2, MAP:0.593, Best MAP: 0.593


class BiHalfModelUnsupervised(nn.Module):
    def __init__(self, d, bit, config=None):
        super(BiHalfModelUnsupervised, self).__init__()
        # TODO: add a layer to match the input size
        self.fc = nn.Linear(d, 512)
        self.fc_encode = nn.Linear(512, bit)
        self.config = config

    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(ctx, U):
            # Yunqiang for half and half (optimal transport)
            _, index = U.sort(0, descending=True)
            N, D = U.shape
            B_creat = torch.cat((torch.ones([int(N / 2), D]), -torch.ones([N - int(N / 2), D]))).to(U.device)
            B = torch.zeros(U.shape).to(U.device).scatter_(0, index, B_creat)
            ctx.save_for_backward(U, B)
            return B

        @staticmethod
        def backward(ctx, g):
            U, B = ctx.saved_tensors
            add_g = (U - B) / (B.numel())
            grad = g + config["gamma"] * add_g
            return grad

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        h = self.fc_encode(x)
        if not self.training:
            return h.sign()
        else:
            b = BiHalfModelUnsupervised.Hash.apply(h)
            target_b = F.cosine_similarity(b[:x.size(0) // 2], b[x.size(0) // 2:])
            target_x = F.cosine_similarity(x[:x.size(0) // 2], x[x.size(0) // 2:])
            loss = F.mse_loss(target_b, target_x)
            return loss


def get_config():
    config = {
        "gamma": 4,
        "optimizer": {"type": optim.Adam, "epoch_lr_decrease": 30,
                      "optim_params": {"lr": 0.001, "weight_decay": 5e-5}},
        "info": "[BiHalf Unsupervised]",
        "batch_size": 128,
        "net": BiHalfModelUnsupervised,
        "epoch": 75,
        "test_map": 5,
        "device": torch.device("mps"),
        "dataset": "siftsmall",
        "bit_list": [14], "topK": 1000
    }

    return config


def train(base_vectors, bit, config=None):
    if config is None:
        config = get_config()
    dataset_loader, num_dataset, dim = prepare_data(base_vectors, config["batch_size"])

    device = config["device"]
    config["num_train"] = num_dataset
    net = config["net"](dim, bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    for epoch in range(config["epoch"]):

        lr = config["optimizer"]["optim_params"]["lr"] * (0.1 ** (epoch // config["optimizer"]["epoch_lr_decrease"]))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, lr:%.9f, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, lr, config["dataset"]), end="")

        net.train()

        train_loss = 0
        for vector in dataset_loader:
            vector = vector.to(device)

            optimizer.zero_grad()

            loss = net(vector)

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(dataset_loader)

        print("\b\b\b\b\b\b\b loss:%.9f" % (train_loss))

    dataset_encoded = []
    net.eval()
    for vector in dataset_loader:
        vector = vector.to(device)
        output = net(vector)
        output[output < 0] = 0
        dataset_encoded.extend(output.data.cpu().numpy().astype(int).tolist())
    return np.array(dataset_encoded)


if __name__ == "__main__":
    config = get_config()
    base_vectors = get_base_data(config["dataset"])
    print(config)
    for bit in config["bit_list"]:
        train(base_vectors, bit, config)
