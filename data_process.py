import json
import numpy as np
import pandas as pd
import pickle
import random

from scipy import sparse
from scipy.sparse import csr_matrix,lil_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange


# 读取CSV文件
# df_satellite_orbit = pd.read_csv('dataset/SO/关系-卫星轨道表.csv')
df_close_encounters = pd.read_csv('dataset/SO/关系-近接卫星表.csv')

with open('dataset/SO/轨道表.csv.json','r') as f:
    content=f.read()
    orbits = json.loads(content)

satellite_to_index = {}
satellites=set()
orbit_to_index = {}
row = []
col = []
values = []
orbit_data=[]

for i in range(len(orbits)):
    # 创建卫星索引映射的字典
    orbit_key = orbits[i]['\ufeffid']
    orbit_to_index[orbit_key] = i
 
    parts = orbits[i]['轨道根二'].split()

    # 提取并处理sat_num
    sat_num = int(parts[1])  # NORAD卫星编号
    sat_key = f"卫星-{sat_num}"
    satellites.add(sat_key)
    
    # 如果该卫星还未在索引映射中，添加进去
    if sat_key not in satellite_to_index:
        satellite_to_index[sat_key] = len(satellite_to_index)


    row.append(satellite_to_index[sat_key]) # sat
    col.append(i) # orbit
    values.append(1)

    # 轨道参数
    inclination = float(parts[2])       # 轨道倾角
    raan = float(parts[3])              # 升交点赤经
    eccentricity = float("0." + parts[4]) # 轨道偏心率
    arg_perigee = float(parts[5])       # 近地点幅角
    mean_anomaly = float(parts[6])      # 平近点角
    mean_motion = float(parts[7][:-1])       # 每天环绕地球的圈数

    orbit_data.append(np.array([
        inclination,
        raan,
        eccentricity,
        arg_perigee,
        mean_anomaly,
        mean_motion
    ]))


orbit_data_array = np.array(orbit_data)

# 创建 MinMaxScaler 对象
scaler = MinMaxScaler()

# 对数据进行归一化
normalized_data = scaler.fit_transform(orbit_data_array)


sat_orbit=sparse.coo_matrix((values, (row, col)), shape=(len(orbits), len(orbits))) #对角

index_to_sat = {i:j for j,i in satellite_to_index.items() }
index_to_orbit = {i:j for j,i in orbit_to_index.items() }

print('read orbit data complete')


# 处理近接事件数据
def process_encounters(encounters_df):
    encounters_dict = {}
    for _, row in encounters_df.iterrows():
        # 根据事件创建一个dict，包含近接与被近接。然后根据type值添加卫星
        if row['source'] not in encounters_dict:
            encounters_dict[row['source']] = {'approached': None, 'approaching': None} 
        if row['TYPE'] == '被接近卫星':
            encounters_dict[row['source']]['approached'] = row['target']
        elif row['TYPE'] == '接近卫星':
            encounters_dict[row['source']]['approaching'] = row['target']
    
    processed_encounters = []
    for event, satellites in encounters_dict.items():
        if satellites['approached'] and satellites['approaching']:
            processed_encounters.append((satellites['approached'], satellites['approaching']))
    
    return pd.DataFrame(processed_encounters, columns=['approached', 'approaching'])

processed_encounters = process_encounters(df_close_encounters) #得到两个卫星近接关系的表。


# 划分数据集,只划分近接事件数据集
train_encounters, test_encounters = train_test_split(processed_encounters, test_size=0.2, random_state=42)

# 根据近接事件表的训练集，构建卫星卫星邻接矩阵。
def create_satellite_satellite_matrix(data):
    row = []
    col = []
    values = []
    for _, row_data in data.iterrows():
        sat1 = satellite_to_index[row_data['approached']]
        sat2 = satellite_to_index[row_data['approaching']]
        row.extend([sat1, sat2]) # 同时添加对称的元素
        col.extend([sat2, sat1])
        values.extend([1, 1])
    return sparse.coo_matrix((values, (row, col)), shape=(len(satellites), len(satellites)))

# 创建训练集矩阵

train_sat_sat = create_satellite_satellite_matrix(train_encounters)
test_sat_sat = create_satellite_satellite_matrix(test_encounters)

print('sat-sat.nnz:',train_sat_sat.nnz)

# 对测试集进行负采样
test=test_sat_sat
train=train_sat_sat

train=train.todok()
test_u=test.row
test_v=test.col
test_data=[]
n=test_u.size

for i in range(n):
    u=test_u[i]
    v=test_v[i]
    test_data.append([u,v])
    #neg sampling
    for _ in range(99):
        j=np.random.randint(test.shape[1])
        while(u,j) in train or j==v:
            j=np.random.randint(test.shape[1])
        test_data.append([u,j])


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import trange

# 假设我们已经提取了轨道参数数据 (n x 6 矩阵)
# 假设我们有卫星近接事件的轨道对 (m x 2 矩阵)##### remark：这里用了train的近接事件。
n = len(orbits)

# 构建输入数据和标签
X = []
y = []

orbit_data = normalized_data

for i in range(n):
    u=test_u[i]
    v=test_v[i]
    X.append(np.concatenate([orbit_data[u], orbit_data[v]]))## 刚好卫星index与轨道index一致
    y.append(1)  # 1 表示这对轨道上发生过近接事件


# 生成负样本（即没有近接事件的轨道对）
num_neg_samples = len(train_encounters)*1
for _ in trange(num_neg_samples):

    i, j = np.random.choice(len(orbit_data), 2, replace=False)
    while (i, j) in train or (j,i) in train or i == j:
        j=np.random.randint(test.shape[1])
    X.append(np.concatenate([orbit_data[i], orbit_data[j]]))
    y.append(0)  # 0 表示这对轨道上没有发生近接事件
        

# 转换为numpy数组
X = np.array(X)
y = np.array(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch的张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


print("starting train orbit-orbit matrix")
# GPU
class OrbitProximityNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(OrbitProximityNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义模型参数
input_dim = X_train.shape[1]  # 输入维度（两个轨道参数的总和）12
hidden_dim = 64  # 隐藏层神经元数量

# 创建模型并转移到 GPU
model = OrbitProximityNet(input_dim, hidden_dim).to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss() #二元交叉熵
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 70
batch_size = 512

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for i in range(0, X_train.size(0), batch_size):
        # 提取批次数据并转移到 GPU
        x_batch = X_train[i:i+batch_size].to(device)
        y_batch = y_train[i:i+batch_size].to(device)

        optimizer.zero_grad()
        outputs = model(x_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')



n = len(orbit_data)
batch_size = 1024  # Set batch size according to GPU memory
threshold = 0.8  # Threshold for determining adjacency 0.5--6933622 0.6--3396018 0.7--1795966 0.8--887656

# Convert orbit_data to a torch tensor and move it to GPU
orbit_data = torch.tensor(orbit_data, dtype=torch.float32).to('cuda')

# Lists to store sparse matrix data
row = []
col = []
data = []

model.eval()
with torch.no_grad():
    for i in range(n):
        # Determine the size of the current batch
        # 这个操作避免了分类讨论
        batch_size_i = min(batch_size, n - i - 1)
        
        if batch_size_i > 0:
            # Prepare the batch for the current orbit i

            ### 这里用了向量化操作，不用在一个batch内，一个一个添加数据。
            batch_data_i = orbit_data[i].repeat(batch_size_i, 1) # 在第i行，都是orbit_data[i]；
            batch_data_j = orbit_data[i + 1:i + 1 + batch_size_i]
            
            # Concatenate i and j data to form input vectors
            input_batch = torch.cat([batch_data_i, batch_data_j], dim=1)
            
            # Perform batch prediction on GPU
            predictions = model(input_batch).view(-1)
            
            # Apply threshold to determine adjacency
            mask = predictions >= threshold
            
            # Get indices where the prediction is above the threshold
            valid_indices = torch.nonzero(mask).view(-1).cpu().numpy()
            
            # Store the results
            row.extend([i] * len(valid_indices))
            col.extend((valid_indices + i + 1).tolist())
            data.extend([1] * len(valid_indices))
    
    # After the loop, mirror the upper triangle to the lower triangle
    row.extend(col)
    col.extend(row[:len(col)])
    data.extend(data)

# Create the sparse matrix
orbit_orbit = sparse.coo_matrix((data, (row, col)), shape=(n, n))

print("orbit-orbit:",orbit_orbit.nnz)


# 距离矩阵的处理
train_sat_sat=(train_sat_sat+ sparse.eye(len(satellites))).tocsr()
orbit_orbit=(orbit_orbit + sparse.eye(len(orbits))).tocsr()
userNum = len(satellites)
itemNum = len(orbits)
trainMat = sat_orbit.tocsr()

UiDistance_mat = (sparse.dok_matrix((userNum+itemNum, userNum+itemNum))).tocsr()
trainMat_T = trainMat.T
for i in range(userNum+itemNum):
    if i < userNum:
        UiDistance_mat[i, userNum:] = trainMat[i]
    else:
        UiDistance_mat[i, :userNum] = trainMat_T[i-userNum]


# 构建metapath
print("start constract Graph")
satellite_satellite = train_sat_sat
satellite_orbit = sat_orbit

# 创建图
G = nx.Graph()

# 添加卫星节点
num_satellites = satellite_satellite.shape[0]
for i in range(num_satellites):
    G.add_node(index_to_sat[i], type='satellite')

# 添加轨道节点
num_orbits = orbit_orbit.shape[0]
for i in range(num_orbits):
    G.add_node(index_to_orbit[i], type='orbit')


# 添加卫星-卫星边（近接关系）
satellite_satellite_coo = satellite_satellite.tocoo()
for i, j in zip(satellite_satellite_coo.row, satellite_satellite_coo.col):
    if i < j:  # 避免重复边
        G.add_edge(index_to_sat[i], index_to_sat[j], type='近接')

# 添加卫星-轨道边（运行关系）
satellite_orbit_coo = satellite_orbit.tocoo()
for i, j in zip(satellite_orbit_coo.row, satellite_orbit_coo.col):
    G.add_edge(index_to_sat[i], index_to_orbit[j], type='运行于')

# 添加轨道-轨道边
orbit_orbit_coo = orbit_orbit.tocoo()
for i, j in zip(orbit_orbit_coo.row, orbit_orbit_coo.col):
    if i < j:  # 避免重复边
        G.add_edge(index_to_orbit[i], index_to_orbit[j], type='邻接')

print(f"Total nodes: {G.number_of_nodes()}")
print(f"Satellite nodes: {sum(1 for _, data in G.nodes(data=True) if data['type'] == 'satellite')}")
print(f"Orbit nodes: {sum(1 for _, data in G.nodes(data=True) if data['type'] == 'orbit')}")
print(f"Total edges: {G.number_of_edges()}")
print(f"近接 edges: {sum(1 for _, _, data in G.edges(data=True) if data['type'] == '近接')}")
print(f"运行于 edges: {sum(1 for _, _, data in G.edges(data=True) if data['type'] == '运行于')}")
print(f"邻接 edges: {sum(1 for _, _, data in G.edges(data=True) if data['type'] == '邻接')}")

def random_walk_metapath(G, metapath, num_samples):
    paths = []
    
    # 获取起始节点类型的节点集合
    start_nodes = [n for n, d in G.nodes(data=True) if d['type'] == metapath[0]]

    for _ in range(num_samples):
        current_node = random.choice(start_nodes)  # 随机选择起始节点
        path = [current_node]
        
        for node_type in metapath[1:]:
            neighbors = [n for n in G.neighbors(current_node) if G.nodes[n]['type'] == node_type]
            
            if not neighbors:  # 如果没有符合条件的邻居节点，则结束当前路径
                break
            
            current_node = random.choice(neighbors)  # 随机选择符合条件的邻居节点
            path.append(current_node)
        
        # 如果路径长度与元路径匹配，则记录该路径
        if len(path) == len(metapath) :
            paths.append(path)
    
    return paths

# 构建 metapath: 随机采样
num_samples=100000
metapath_1 = ['satellite', 'orbit', 'orbit', 'orbit', 'satellite']
metapath_2 = ['orbit', 'satellite', 'satellite', 'satellite', 'orbit']

mp_sooos = random_walk_metapath(G, metapath_1, num_samples)
mp_ossso = random_walk_metapath(G, metapath_2, num_samples)
print('Random sampling hyperedges_satellite complete')
print('SOOOS:',len(mp_sooos))
print('OSSSO:',len(mp_ossso))

# 构建 hyperedge
def build_sparse_incidence_matrices(metapath_instances, metapath):
    # 创建节点到索引的映射
    node_to_index={'satellite':satellite_to_index,'orbit': orbit_to_index}
    hyperedges = defaultdict(list)
    
    # 遍历metapath实例,构建超边
    for instance in metapath_instances:
        for i in range(len(instance) - 1):
            if i<np.ceil(len(instance)/2.0):
                source_type,target_type = metapath[i:i+2]
                source = node_to_index[source_type][instance[i]]
                target = node_to_index[target_type][instance[i+1]]
            else:
                target_type,source_type = metapath[i:i+2]
                source = node_to_index[source_type][instance[i+1]]
                target = node_to_index[target_type][instance[i]]

            hyperedges[(source_type, target_type)].append((source, target))

    # 构建稀疏incidence矩阵
    incidence_matrices = {}
    for (source_type, target_type), edges in hyperedges.items():
        row = [edge[0] for edge in edges]
        col = [edge[1] for edge in edges]
        data = np.ones(len(edges))
        shape = (len(node_to_index[source_type]), len(node_to_index[target_type]))
        matrix = csr_matrix((data, (row, col)), shape=shape)
        incidence_matrices[(source_type, target_type)] = matrix
    
    return incidence_matrices


# 处理两种metapath, 生成hyperedge
h_SOOOS = build_sparse_incidence_matrices(mp_sooos, metapath_1)
h_OSSSO = build_sparse_incidence_matrices(mp_ossso, metapath_2)

for (source_type, target_type), matrix in h_SOOOS.items():

    print(f"Matrix {source_type}-{target_type}:")
    print(f"  Shape: {matrix.shape}")
    print(f"  Non-zero elements: {matrix.nnz}")

for (source_type, target_type), matrix in h_OSSSO.items():

    print(f"Matrix {source_type}-{target_type}:")
    print(f"  Shape: {matrix.shape}")
    print(f"  Non-zero elements: {matrix.nnz}")

# 构建 H 矩阵
H_SS = h_OSSSO[('satellite','satellite')]
H_OO = h_SOOOS[('orbit','orbit')]

total = userNum+itemNum
newMatrix = sparse.dok_matrix((total, total))

for i in range(userNum):
    non_zero = h_SOOOS[('satellite','orbit')][i].nonzero()[1]
    for j in non_zero:
        newMatrix[i, userNum + j] = 1  

for i in range(itemNum):
    non_zero = h_OSSSO[('orbit','satellite')][i].nonzero()[1]
    for j in non_zero:
        newMatrix[userNum + i, j] = 1  

H_SO = newMatrix.tocsr()


# 储存数据
with open('dataset/SO/satellite_index_mapping.pkl', 'wb') as f:
    pickle.dump(satellite_to_index, f)

with open('dataset/SO/orbit_index_mapping.pkl', 'wb') as f:
    pickle.dump(orbit_to_index, f)

distanceMat=(train_sat_sat,orbit_orbit,UiDistance_mat,sat_orbit)
with open('dataset/SO/distanceMat.pkl', 'wb') as fs:
    pickle.dump(distanceMat, fs)

data = (train_sat_sat,test_data)
with open("dataset/SO/data.pkl", 'wb') as fs:
    pickle.dump(data, fs)

metapaths=(mp_sooos,mp_ossso)
with open("dataset/SO/metapaths.pkl", 'wb') as fs:
    pickle.dump(metapaths, fs)

hyperegdes=(H_SS,H_OO,H_SO)
with open("dataset/SO/hyperedges.pkl", 'wb') as fs:
    pickle.dump(hyperegdes, fs)

print("所有矩阵、映射和测试数据已保存。")