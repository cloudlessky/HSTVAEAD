import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from util.time import *
from util.env import *
from sklearn.metrics import mean_squared_error
from util.data import *
from torch.utils.data import DataLoader, random_split, Subset
from torch import optim
from EarlyStopping import EarlyStopping
import random

torch.set_default_tensor_type(torch.FloatTensor)
np.set_printoptions(precision=5, suppress=True)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=50)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def point_adjust_eval(anomaly_start, anomaly_end, down, loss, thr1, thr2):
    len_ = loss.shape
    win_len = 40
    anomaly = np.zeros((len_))
    anomaly[np.where(loss>thr1)] = 1
    anomaly[np.where(loss<thr2)] = 1
    ground_truth = np.zeros((len_))
    for i in range(len(anomaly_start)):
        ground_truth[int(anomaly_start[i]/down)- win_len :int(anomaly_end[i]/down)- win_len +1] = 1
        if np.sum(anomaly[int(anomaly_start[i]/down)- win_len :int(anomaly_end[i]/down)- win_len])>0:
            anomaly[int(anomaly_start[i]/down)- win_len :int(anomaly_end[i]/down)- win_len] = 1
        anomaly[int(anomaly_start[i]/down)- win_len ] = ground_truth[int(anomaly_start[i]/down)- win_len ]
        anomaly[int(anomaly_end[i]/down)- win_len] = ground_truth[int(anomaly_end[i]/down)- win_len]
    return anomaly, ground_truth

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def calculate_kl_loss(z_mean, z_log_sigma):
    temp = 1.0 + 2 * z_log_sigma - z_mean ** 2 - torch.exp(2 * z_log_sigma)
    return -0.5 * torch.sum(temp, 1)

def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')
    return loss

def train(model = None, device=None,save_path='', config={}, train_dataloader=None, val_dataloader=None,group_index=1,index=1):
    N_EPOCHS = config['epoch']
    min_loss = 1e+8
    train_loader = train_dataloader
    valid_loader = val_dataloader
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    es = EarlyStopping(patience=5)
    swap_count=0
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        model.train()
        batch_losses = 0
        batch_i = 0
        for all_input, all_target_seq, attack_labels in train_loader:
            optimizer.zero_grad()
            recon,pred,batch_loss,prob,spatio_error,batch_seq_loss_list = model(all_input, all_target_seq, batch_i,config,1,device)
            batch_i += 1
            batch_losses += batch_loss
            batch_loss.backward()
            optimizer.step()
        epoch_loss = batch_losses / batch_i

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print('Epoch: {epoch} | Time: {epoch_mins}m {epoch_secs}s'.format(epoch=epoch,
                                                                          epoch_mins=epoch_mins,
                                                                          epoch_secs=epoch_secs))
        print('\tTrain Loss: {train_loss:.3f}'.format(train_loss=epoch_loss))
        f = open('./loss/train_loss.txt', 'a')
        print('epoch:', epoch, 'group_index,index',group_index,index,' train_loss:', epoch_loss, 'train time',epoch_mins,epoch_secs,file=f)
        f.close()

        valid_loss = val(model,valid_loader,config, device)
        valid_loss_cpu=valid_loss.cpu()

        if es.step(valid_loss_cpu):
            print("Early Stopping")
            break

        if valid_loss < min_loss:
            swap_count+=1
            torch.save(model.state_dict(), save_path)
            min_loss = valid_loss
        print('swap_count',swap_count)
        print('\t Val. Loss: {valid_loss}'.format(valid_loss=valid_loss))

def val(model,testloader,config, device):
    model.eval()
    acu_loss = 0
    batch_i=0
    for batch_input, batch_target, attack_labels in testloader:
        batch_i += 1
        with torch.no_grad():
            recon,pred,batch_losses,prob,spatio_error,batch_seq_loss_list = model(batch_input, batch_target, batch_i, config,0, device)
        acu_loss+=batch_losses
    avg_loss = acu_loss / batch_i
    return avg_loss

def test(model, testloader, config, device):
    model.eval()
    N = config['num_tasks']
    batch_size = config['batch']
    down = config['down']
    test_loss_list = []
    seq_loss_list = []
    prob_tensor = torch.zeros((batch_size, N * N, 2))
    batch_i=0
    for batch_input, batch_target, attack_labels in testloader:
        with torch.no_grad():
            recon,pred_all,batch_losses,prob,spatio_error,batch_seq_loss_list = model(batch_input, batch_target, batch_i, config, 0, device)

            batch_i+=1
            if batch_i == 1:
                seq_loss_list = batch_seq_loss_list
            else:
                for i in range(len(batch_seq_loss_list)):
                    seq_loss_list[i] += batch_seq_loss_list[i]
            pred = pred_all[:, :, 0]
            y_pred = batch_target[:,:,0]
            pred_future = pred_all[:, :, 1:]
            y_pred_future = batch_target[:, :, 1:]
            recon_history = recon
            y_recon_history = batch_input
            spatio_err = spatio_error.unsqueeze(1).repeat(1, pred.shape[1])
            labels = attack_labels.unsqueeze(1).repeat(1, pred.shape[1])
            prob_tensor = torch.cat([prob_tensor, prob.cpu()], dim=0)
            if batch_i == 1:
                t_test_pre = pred
                t_test_trg = y_pred

                t_test_pre_future = pred_future
                t_test_trg_future = y_pred_future

                t_test_rec_history = recon_history
                t_test_trg_history = y_recon_history
                t_test_labels = labels

                t_test_spatioerr = spatio_err
            else:
                t_test_pre = torch.cat((t_test_pre, pred), dim=0)
                t_test_trg = torch.cat((t_test_trg, y_pred), dim=0)
                t_test_pre_future = torch.cat((t_test_pre_future, pred_future), dim=0)
                t_test_trg_future = torch.cat((t_test_trg_future, y_pred_future), dim=0)
                t_test_rec_history = torch.cat((t_test_rec_history, recon_history), dim=0)
                t_test_trg_history = torch.cat((t_test_trg_history, y_recon_history), dim=0)
                t_test_labels = torch.cat((t_test_labels, labels), dim=0)
                t_test_spatioerr = torch.cat((t_test_spatioerr, spatio_err), dim=0)
        test_loss_list.append(batch_losses.item())

    mat_test = prob_tensor.reshape(prob_tensor.shape[2], prob_tensor.shape[0], N, N).cpu()

    w = 2
    anomaly_time = pd.read_csv('../data/swat/SWAT_Time.csv')
    anomaly_time = np.array(anomaly_time.iloc[:, :2])
    anomaly_start = anomaly_time[:, 0]
    anomaly_end = anomaly_time[:, 1]

    total_out_degree_move_filtered = np.zeros((mat_test.shape[1], N))
    for fe in range(N):
        y = (torch.mean(mat_test[0, :, :, fe], -1))
        xx = moving_average(y, w)
        total_out_degree_move_filtered[1:, fe] = y[w - 1:] - xx
        total_out_degree_move_filtered[0, fe] = 0

    loss_degree = np.mean(total_out_degree_move_filtered, 1)

    f1 = np.zeros((200, 200))
    thr1_mat = np.zeros((200, 200))
    thr2_mat = np.zeros((200, 200))
    loss_degree_max = np.max(loss_degree)
    loss_degree_min = np.min(loss_degree)
    mid = (loss_degree_max + loss_degree_min) / 2
    step1 = (mid - loss_degree_min) / 200
    step2 = (loss_degree_max - mid) / 200

    i = 0
    for thr1 in np.arange(mid, loss_degree_max, step2):
        j = 0
        for thr2 in np.arange(loss_degree_min, mid, step1):
            if i < 200 and j < 200:
                anomaly, ground_truth = point_adjust_eval(anomaly_start, anomaly_end, down, (loss_degree), thr1,thr2)
                f1[i, j] = f1_score(ground_truth, anomaly)
                thr1_mat[i, j] = thr1
                thr2_mat[i, j] = thr2
                j += 1
        i += 1

    pos = np.unravel_index(np.argmax(f1), f1.shape)
    print('pos', pos)

    anomaly, ground_truth = point_adjust_eval(anomaly_start, anomaly_end, down, (loss_degree),
                                              thr1_mat[pos[0], pos[1]],
                                              thr2_mat[pos[0], pos[1]])

    test_pre_list = t_test_pre.tolist()
    test_trg_list = t_test_trg.tolist()
    test_pre_future = t_test_pre_future.tolist()
    test_trg_future = t_test_trg_future.tolist()
    test_rec_history = t_test_rec_history.tolist()
    test_trg_history = t_test_trg_history.tolist()
    test_labels_list = t_test_labels.tolist()
    test_spatioerr_list = t_test_spatioerr.tolist()

    avg_loss = sum(test_loss_list) / len(test_loss_list)
    return avg_loss, [test_pre_list, test_trg_list, test_labels_list, test_spatioerr_list],[test_pre_future, test_trg_future, test_rec_history, test_trg_history],anomaly

