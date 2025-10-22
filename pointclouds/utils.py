import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

def sample_pointclouds(xtrain, ytrain, xval, yval, classes, num_pointclouds, num_pointclouds_test):
    xtrain_s = []
    ytrain_s = []
    xtest_s = []
    ytest_s = []
    idxs_out =[]
    idxs_out_test = []
    for i in classes:
        indices = [s for s in range(len(ytrain)) if ytrain[s]==i]
        indices_test = [s for s in range(len(yval)) if yval[s]==i]
        #indices = slice(*idxs[0:num_pointclouds])
        if num_pointclouds == -1 or num_pointclouds>len(indices):
            num_pointclouds = len(indices)
        if num_pointclouds_test == -1 or num_pointclouds_test>len(indices_test):
            num_pointclouds_test = len(indices_test)
        for s in range(num_pointclouds):
            xtrain_s.append(xtrain[indices[s]])
            ytrain_s.append(ytrain[indices[s]])
            idxs_out = indices[s]
        for s in range(num_pointclouds_test):
            xtest_s.append(xval[indices_test[s]])
            ytest_s.append(yval[indices_test[s]])
            idxs_out_test = indices_test[s]
    return xtrain_s, ytrain_s, xtest_s, ytest_s, idxs_out, idxs_out_test

def get_fs(X):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  '''
  input: X (n by 3)
  output: Data (class) storing fs (n by (n + n(n-1)/2 + 1)) feature matrix
   f_d is the set of diagonal edge attributes,
   f_o is the set of off-diagonal (upper) edge attributes,
   f_star is \sum_{i \neq j} X_ii X_ij where X is the n by n edge attribute graph
  '''
  n = X.shape[0]
  data = Data()
  Gram = torch.FloatTensor(X @ X.T) #🌟🌟🌟🌟🌟
  # off_mask = torch.triu(torch.ones(n, n)) == 1
  off_mask = ~torch.eye(n, dtype=bool)

  data.f_d = torch.diagonal(Gram, 0).unsqueeze(1)
  data.f_o = Gram[off_mask].unsqueeze(1)
  data.f_star = (data.f_d * (torch.sum(Gram, dim=1, keepdim=True) - data.f_d)).sum().reshape((1,1))

  row = torch.arange(n).repeat_interleave(n)
  col = torch.arange(n).repeat(n)
  data.edge_index = torch.stack([row, col], dim=0).to(device)

  data.edge_attr = Gram.flatten().to(device)
  data.x = torch.FloatTensor(X) 

  return data

def binary_expansion(f, num_radial, theta=1):
  '''
  input: f (bs x 1); num_radial - number of basis expansion
  output: phi(f) (bs x num_radial)
  phi(f) = [..., sigmoid(f-theta/theta), sigmoid(f/theta), sigmoid(f+theta/thera),...]
  '''
  bs = f.shape[0]
  out = torch.zeros((bs, num_radial))
  max_val = (num_radial - 1)//2
  offsets = np.arange(start=-max_val, stop=max_val+1) #symmetric construction
  for i, offset in enumerate(offsets):
    out[:, i:(i+1)] = F.sigmoid( (f - theta*offset) / theta )
  return out

def get_binary_expansion(data, num_radial=100, theta=1):
  '''
  Apply binary_expansion for all fs
  try: num_radial \in {100, 1000}
  '''
  #print(f"num_basis={num_radial}")
  data.f_d = binary_expansion(data.f_d, num_radial, theta)
  data.f_o = binary_expansion(data.f_o, num_radial, theta)
  data.f_star = binary_expansion(data.f_star, num_radial, theta)
  return data 

def get_Kmeans(X, k=3):
  '''
  Implement eqn 6: replace f_d as the self-dots of the K-means dots
  input: X (n by 3), k: number of k-means clusters
  output: Data (class) storing  feature matrix
   f_d is the flattened feature of the gram matrix of K-means centroids
   f_o is the set of dot products of each point to the K-means centroids
  '''
  n = X.shape[0]
  data = Data()

  kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++', n_init=1).fit(X)
  K = kmeans.cluster_centers_ #(num_centroids, 3)
  #sort Kmeans centroid by norm
  indexlist = np.argsort(np.linalg.norm(K,axis=1))
  K = K[indexlist,:]
  data.f_o = torch.FloatTensor( X @ K.T) #shape(n, d)
  Gram_k = torch.FloatTensor( K @ K.T ) #shape (d, d)
  data.f_d = Gram_k.reshape(1, k*k) #shape (d^2,)
  return data

def get_subgraph(X, r):
    data = Data()
    data.pos = torch.FloatTensor(X)  # [N, 3]

    tree = KDTree(X)
    edge_list = []
    aggregated_relative = []  # [N, 3]

    for i, center in enumerate(X):
        idx = tree.query_radius(center.reshape(1, -1), r=r)[0]
        
        for j in idx:
            if i != j:
                edge_list.append([i, j])
        
        idx2 = tree.query_radius(center.reshape(1, -1), r=100)[0]
        neighbors = X[idx2]
        relative_pos = neighbors - center  # [K_i, 3]

        if len(relative_pos) > 0:
            pooled = torch.max(torch.tensor(relative_pos, dtype=torch.float32), dim=0)[0]  # [3]
        else:
            pooled = torch.zeros(3)

        aggregated_relative.append(pooled)

    data.edge_index = torch.tensor(edge_list).T  # shape: [2, E]
    data.h = torch.stack(aggregated_relative, dim=0)  # shape: [N, 3]

    # edge_attr 可选：你现在也能保留 Gram 特征作为边的属性
    Gram = torch.FloatTensor(X @ X.T)
    row = data.edge_index[0]
    col = data.edge_index[1]
    data.edge_attr = Gram[row, col]  # shape: [E]

    return data

def build_dataset(xtrain_s, xtest_s, ids, Kmeans=False, k=3, r=0.3):
  X_train = []
  X_test = []
  for i in ids:
    if Kmeans:
      X_train.append(get_Kmeans(xtrain_s[i],k=k))
      X_test.append(get_Kmeans(xtest_s[i],k=k))
    else:
      # X_train.append(get_binary_expansion(get_fs(xtrain_s[i])))
      # X_test.append(get_binary_expansion(get_fs(xtest_s[i])))
      # X_train.append(get_subgraph(xtrain_s[i], r=r))
      # X_test.append(get_subgraph(xtest_s[i], r=r))
      X_train.append(get_invariant_graph_data(xtrain_s[i], r=r))
      X_test.append(get_invariant_graph_data(xtest_s[i], r=r))
      

  return X_train, X_test

def get_loaders(train_data1, test_data1, train_data2, test_data2, batch_size, follow_batch=None):
  '''
  data1, data2 refers to subsets in ModelNet10 belonging to two different classes
  '''

  train_loader1 = DataLoader(
      train_data1,
      batch_size=batch_size,
      shuffle=False,
      follow_batch=follow_batch
  )
  train_loader2 = DataLoader(
      train_data2,
      batch_size=batch_size,
      shuffle=False,
      follow_batch=follow_batch
  )

  test_loader1 = DataLoader(
      test_data1,
      batch_size=batch_size,
      shuffle=False,
      follow_batch=follow_batch
  )

  test_loader2 = DataLoader(
      test_data2,
      batch_size=batch_size,
      shuffle=False,
      follow_batch=follow_batch
  )

  return train_loader1, train_loader2, test_loader1, test_loader2

def extract_values(file_path):
    values = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if line.strip() and not line.startswith('r_value'):
            parts = line.strip().split()
            ari_values = list(map(float, parts[1].split(',')))
            values.append(ari_values)

    return values
  
def get_invariant_graph_data(X, r):
    
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
        
    data = Data()
    data.pos = torch.from_numpy(X).float()

    tree = KDTree(X)

    edge_indices = tree.query_radius(X, r=r)
    edge_list = []
    for i, neighbors_of_i in enumerate(edge_indices):
        for j in neighbors_of_i:
            if i != j:
                edge_list.append([i, j])
    data.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    distances, _ = tree.query(X, k=100)
    k_distances = torch.from_numpy(distances[:, 1:]).float() # [N, k]
    
    mean_dist = torch.mean(k_distances, dim=1) # [N]
    max_dist = torch.max(k_distances, dim=1).values # [N]
    min_dist = torch.min(k_distances, dim=1).values # [N]
    std_dist = torch.std(k_distances, dim=1)   # [N]
    
    # std_dist = torch.nan_to_num(std_dist, nan=0.0)
    # data.h = torch.ones((X.shape[0], 1), dtype=torch.float32)
    data.h = torch.stack([mean_dist, max_dist, min_dist, std_dist], dim=1)

    X_torch = data.pos
    x_norm_sq = torch.sum(X_torch**2, dim=1, keepdim=True)
    dist_sq = x_norm_sq - 2 * torch.mm(X_torch, X_torch.t()) + x_norm_sq.t()

    dist_sq.fill_diagonal_(0)
    dist_sq = F.relu(dist_sq) 
    dist_matrix = torch.sqrt(dist_sq)

    adj_mask_full = torch.ones_like(dist_matrix, dtype=torch.bool)
    adj_mask_full.fill_diagonal_(False)
    
    mask_A1 = (dist_matrix <= 0.1) & adj_mask_full
    mask_A2 = (dist_matrix > 0.1) & (dist_matrix <= 0.7) & adj_mask_full

    row1, col1 = torch.nonzero(mask_A1, as_tuple=True)
    data.edge_index_A1 = torch.stack([row1, col1], dim=0)
    
    row2, col2 = torch.nonzero(mask_A2, as_tuple=True)
    data.edge_index_A2 = torch.stack([row2, col2], dim=0)


    Gram = torch.FloatTensor(X @ X.T)
    row = data.edge_index[0]
    col = data.edge_index[1]
    data.edge_attr = Gram[row, col]  # shape: [E]

    return data