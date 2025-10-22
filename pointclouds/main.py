import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torchmetrics.regression import SpearmanCorrCoef

from utils import sample_pointclouds, build_dataset, get_loaders
from models import PointGNN, SiameseRegressor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import pickle
import random
import copy
import time
from tqdm import tqdm

#cf: metric learning https://en.wikipedia.org/wiki/Similarity_learning

def train(model, optimizer, criterion, loader1, loader2, dist_true):
    model.train()

    total_loss = 0
    for data1, data2 in zip(loader1, loader2):
        data1 = data1.to(device)
        data2 = data2.to(device)
        optimizer.zero_grad()  # Clear gradients.
        dist_pred = model(data1, data2)  # Forward pass.
        loss = criterion(dist_pred, dist_true)  # Loss computation.
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        total_loss += loss.item()

    return total_loss / len(loader1)

@torch.no_grad()
def test(model, criterion, loader1, loader2, dist_true):
    model.eval()

    total_loss = 0
    for data1, data2 in zip(loader1, loader2):
        data1 = data1.to(device)
        data2 = data2.to(device)
        dist_pred = model(data1, data2)  # Forward pass.
        loss = criterion(dist_pred, dist_true)  # Loss computation.
        total_loss += loss.item()

    return total_loss / len(loader1)

def evaluate_dist_mat(model,loader1, loader2):
  model.eval()
  data1 = next(iter(loader1)).to(device)
  data2 = next(iter(loader2)).to(device)
  dist_vec = model(data1, data2).detach().cpu()
  n = data1.num_graphs
  dist_mat = torch.unflatten(dist_vec, 0, (n, n))
  return dist_mat

if __name__ == "__main__":

    data = pickle.load(open(f"Data_PointClouds/ModelNet_np_withLabels_pts=100.pkl", "rb"))
    classes = [2,7]
    num_pointclouds=40
    epsilon = 0.01
    xtrain_s, ytrain_s, xtest_s, ytest_s, indices, indices_test = \
        sample_pointclouds(data['train_pos'], data['train_y'], data['val_pos'], data['val_y'], classes, num_pointclouds, num_pointclouds)
 

    n_train = len(xtrain_s)
    n_test = len(xtest_s)
    bs = 40

    dist_true_train = pickle.load(open("./Data_PointClouds/LB_GW_train.pkl", "rb"))
    dist_true_train = torch.from_numpy(np.array(dist_true_train)).unsqueeze(1)
    dist_true_test = pickle.load(open("./Data_PointClouds/LB_GW_test.pkl", "rb"))
    dist_true_test = torch.from_numpy(np.array(dist_true_test)).unsqueeze(1)

    idx1 = np.where(np.array(ytrain_s)==2)[0]
    idx2 = np.where(np.array(ytrain_s)==7)[0]


    model_name = "PointGNN"
    r_values = [0.06, 0.08] + np.arange(0.1, 1.01, 0.1).tolist() + [1.2, 1.4]
    r_values = [round(r, 2) for r in r_values]

    num_runs = 50
    node_feature_dim = 16
    hid_dim = 16
    out_dim = 16
    
    results_train = []
    results_test=[]
    results_time=[]

    for r in r_values:
        torch.manual_seed(901)
        random.seed(901)
        np.random.seed(901)
        trainc = []
        testc = []
        t = []
        print(f"\033[1;35mRunning for r={r:.2f}\033[0m")

        X_train1, X_test1 = build_dataset(xtrain_s, xtest_s, idx1, r=r)
        X_train2, X_test2 = build_dataset(xtrain_s, xtest_s, idx2, r=r)
        train_loader1, train_loader2, test_loader1, test_loader2 = get_loaders(
            X_train1, X_test1, X_train2, X_test2, batch_size=bs, follow_batch=['f_d','f_o']
        )

        for run in tqdm(range(num_runs), desc=f"r = {r:.1f}"):
            torch.manual_seed(run)
            random.seed(run)
            np.random.seed(run)
            start_time = time.time()

            gnn = PointGNN(node_feature_dim, hid_dim, out_dim)
            model = SiameseRegressor(gnn).to(device)
            model.SM.reset_parameters()
            optimizer = torch.optim.Adam([
                {'params': model.SM.parameters(), 'lr': 1e-2},
                {'params': model.metric.parameters(), 'lr': 1e-2},
                {'params': model.linear.parameters(), 'lr': 1e-2},
            ], lr=1e-3)

            criterion = torch.nn.MSELoss()
            best_test_loss = float('inf')
            best_model_state = None       

            for epoch in range(1, 201):
                train_loss = train(model, optimizer, criterion, train_loader1, train_loader2, dist_true_train.to(device))
                test_loss = test(model, criterion, test_loader1, test_loader2, dist_true_test.to(device))

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_model_state = copy.deepcopy(model.state_dict()) 
                if epoch %10 == 0:
                    print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
            model.load_state_dict(best_model_state)

            D_pred_train = evaluate_dist_mat(model, train_loader1, train_loader2)
            D_pred_test = evaluate_dist_mat(model, test_loader1, test_loader2)

            spearman = SpearmanCorrCoef()
            rank_corr_train = spearman(D_pred_train.reshape(-1,1), dist_true_train).item()
            rank_corr_test = spearman(D_pred_test.reshape(-1,1), dist_true_test).item()
            
            running_time = time.time() - start_time

            print(f"Run {run+1:02d}: Train Corr = {rank_corr_train:.4f}, Test Corr = {rank_corr_test:.4f}, Time = {running_time:.2f}s")
            trainc.append(rank_corr_train)
            testc.append(rank_corr_test)
            t.append(running_time)

        results_train.append(trainc)
        results_test.append(testc)
        results_time.append(t)

    train_means = [np.mean(res) for res in results_train]
    train_stderr = [np.std(res, ddof=1) / np.sqrt(len(res)) for res in results_train]
    test_means = [np.mean(res) for res in results_test]
    test_stderr = [np.std(res, ddof=1) / np.sqrt(len(res)) for res in results_test]
    time_means = [np.mean(res) for res in results_time]
    time_stderr = [np.std(res, ddof=1) / np.sqrt(len(res)) for res in results_time]

    plt.figure(figsize=(10, 6))
    plt.errorbar(r_values, train_means, yerr=train_stderr, fmt='-o', label=model_name, capsize=5)
    plt.xlabel('r (radius)', fontsize=12)
    plt.ylabel('Rank correlation', fontsize=12)
    plt.title(f'Performance of {model_name} (training)', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}/{model_name}_train.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.errorbar(r_values, test_means, yerr=test_stderr, fmt='-o', label=model_name, capsize=5)
    plt.xlabel('r (radius)', fontsize=12)
    plt.ylabel('Rank correlation', fontsize=12)
    plt.title(f'Performance of {model_name} (test)', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}/{model_name}_test.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.errorbar(r_values, time_means, yerr=time_stderr, fmt='-o', label=model_name, capsize=5)
    plt.xlabel('r (radius)', fontsize=12)
    plt.ylabel('Time', fontsize=12)
    plt.title(f'Running time of {model_name}', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}/{model_name}_time.png", dpi=300, bbox_inches='tight')
    plt.show()

    with open(f"results_PointClouds/{model_name}/results_train.txt", "a") as file:
        file.write("r_value\trank_corr_train\n")
        for i, r in enumerate(r_values):
            file.write(f"r={r:.2f}\t{','.join(map(str, results_train[i]))}\n")

    with open(f"results_PointClouds/{model_name}/results_test.txt", "a") as file:
        file.write("r_value\trank_corr_test\n")
        for i, r in enumerate(r_values):
            file.write(f"r={r:.2f}\t{','.join(map(str, results_test[i]))}\n")

    with open(f"results_PointClouds/{model_name}/results_time.txt", "a") as file:
        file.write("r_value\ttime\n")
        for i, r in enumerate(r_values):
            file.write(f"r={r:.2f}\t{','.join(map(str, results_time[i]))}\n")