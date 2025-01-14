import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np


### HGCL
class MODEL(nn.Module):
    def __init__(self,args, userNum, itemNum, userMat, itemMat, uiMat, H_SS, H_OO, H_SO, hide_dim, Layers):
        super(MODEL, self).__init__()
        self.args = args
        self.userNum = userNum
        self.itemNum = itemNum
        self.uuMat = userMat
        self.iiMat = itemMat
        self.uiMat = uiMat
        self.H_SS = H_SS
        self.H_OO = H_OO
        self.H_SO = H_SO
        self.hide_dim = hide_dim
        self.LayerNums = Layers
        
        uimat   = self.uiMat[: self.userNum,  self.userNum:]
        values  = torch.FloatTensor(uimat.tocoo().data)
        indices = np.vstack(( uimat.tocoo().row,  uimat.tocoo().col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape =  uimat.tocoo().shape
        uimat1=torch.sparse.FloatTensor(i, v, torch.Size(shape))
        self.uiadj = uimat1
        self.iuadj = uimat1.transpose(0,1)
        
        self.gating_weightub=nn.Parameter(
            torch.FloatTensor(1,hide_dim))
        nn.init.xavier_normal_(self.gating_weightub.data)
        self.gating_weightu=nn.Parameter( 
            torch.FloatTensor(hide_dim,hide_dim))
        nn.init.xavier_normal_(self.gating_weightu.data)
        self.gating_weightib=nn.Parameter( 
            torch.FloatTensor(1,hide_dim))
        nn.init.xavier_normal_(self.gating_weightib.data)
        self.gating_weighti=nn.Parameter(
            torch.FloatTensor(hide_dim,hide_dim))
        nn.init.xavier_normal_(self.gating_weighti.data)

        self.encoder = nn.ModuleList()
        for i in range(0, self.LayerNums):
            self.encoder.append(GCN_layer())
        self.k = args.rank 
        k = self.k
        self.mlp  = MLP(hide_dim,hide_dim*k,hide_dim//2,hide_dim*k)
        self.mlp1 = MLP(hide_dim,hide_dim*k,hide_dim//2,hide_dim*k)
        self.mlp2 = MLP(hide_dim,hide_dim*k,hide_dim//2,hide_dim*k)
        self.mlp3 = MLP(hide_dim,hide_dim*k,hide_dim//2,hide_dim*k)
        self.meta_netu = nn.Linear(hide_dim*3, hide_dim, bias=True)
        self.meta_neti = nn.Linear(hide_dim*3, hide_dim, bias=True)

        self.embedding_dict = nn.ModuleDict({
        'uu_emb': torch.nn.Embedding(userNum, hide_dim).cuda(),
        'ii_emb': torch.nn.Embedding(itemNum, hide_dim).cuda(),
        'user_emb': torch.nn.Embedding(userNum , hide_dim).cuda(),
        'item_emb': torch.nn.Embedding(itemNum , hide_dim).cuda(),
        })

    def init_weight(self, userNum, itemNum, hide_dim):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(t.empty(userNum, hide_dim))),
            'item_emb': nn.Parameter(initializer(t.empty(itemNum, hide_dim))),
        })
        return embedding_dict

    
    def sparse_matrix_add(self, csr1, csr2):
        # 确保矩阵维度相同
        if csr1.shape != csr2.shape:
            raise ValueError("Matrices must have the same dimensions")
        
        # 执行逻辑或操作
        result = csr1 + csr2
        result.data = np.ones_like(result.data)  # 将所有非零元素设为1
        result.eliminate_zeros()  # 移除由于加法可能产生的零元素
        
        return result
    


    def self_gatingu(self,em):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em,self.gating_weightu) + self.gating_weightub))
    def self_gatingi(self,em):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em,self.gating_weighti) + self.gating_weightib))


    def metafortansform(self, auxiembedu,targetembedu,auxiembedi,targetembedi):
       
        # Neighbor information of the target node
        uneighbor=t.matmul( self.uiadj.cuda(),self.ui_itemEmbedding)
        ineighbor=t.matmul( self.iuadj.cuda(),self.ui_userEmbedding)

        # Meta-knowlege extraction
        tembedu=(self.meta_netu(t.cat((auxiembedu,targetembedu,uneighbor),dim=1).detach()))
        tembedi=(self.meta_neti(t.cat((auxiembedi,targetembedi,ineighbor),dim=1).detach()))
        
        """ Personalized transformation parameter matrix """
        # Low rank matrix decomposition
        metau1=self.mlp( tembedu). reshape(-1,self.hide_dim,self.k)# d*k
        metau2=self.mlp1(tembedu). reshape(-1,self.k,self.hide_dim)# k*d
        metai1=self.mlp2(tembedi). reshape(-1,self.hide_dim,self.k)# d*k
        metai2=self.mlp3(tembedi). reshape(-1,self.k,self.hide_dim)# k*d
        meta_biasu =(torch.mean( metau1,dim=0))
        meta_biasu1=(torch.mean( metau2,dim=0))
        meta_biasi =(torch.mean( metai1,dim=0))
        meta_biasi1=(torch.mean( metai2,dim=0))
        low_weightu1=F.softmax( metau1 + meta_biasu, dim=1)
        low_weightu2=F.softmax( metau2 + meta_biasu1,dim=1)
        low_weighti1=F.softmax( metai1 + meta_biasi, dim=1)
        low_weighti2=F.softmax( metai2 + meta_biasi1,dim=1)

        # The learned matrix as the weights of the transformed network
        tembedus = (t.sum(t.multiply( (auxiembedu).unsqueeze(-1), low_weightu1), dim=1))# Equal to a two-layer linear network; Ciao and Yelp data sets are plus gelu activation function
        tembedus =  t.sum(t.multiply( (tembedus)  .unsqueeze(-1), low_weightu2), dim=1)
        tembedis = (t.sum(t.multiply( (auxiembedi).unsqueeze(-1), low_weighti1), dim=1))
        tembedis =  t.sum(t.multiply( (tembedis)  .unsqueeze(-1), low_weighti2), dim=1)
        transfuEmbed = tembedus
        transfiEmbed = tembedis
        return transfuEmbed, transfiEmbed
    
    # def forward(self, iftraining, uid, iid, norm = 1):
    def forward(self, iftraining, norm = 1):
        
        item_index=np.arange(0,self.itemNum)
        user_index=np.arange(0,self.userNum)
        ui_index = np.array(user_index.tolist() + [ i + self.userNum for i in item_index])
        
        # Initialize Embeddings
        userembed0 = self.embedding_dict['user_emb'].weight
        itemembed0 = self.embedding_dict['item_emb'].weight
        uu_embed0  = self.self_gatingu(userembed0)
        ii_embed0  = self.self_gatingi(itemembed0)
        self.ui_embeddings       = t.cat([ userembed0, itemembed0], 0)
        self.all_user_embeddings = [uu_embed0]
        self.all_item_embeddings = [ii_embed0]
        self.all_ui_embeddings   = [self.ui_embeddings]
        # Encoder

        self.UU = self.sparse_matrix_add(self.uuMat, self.H_SS)
        self.UI = self.sparse_matrix_add(self.uiMat, self.H_SO) ## uiMat: u+i x u+i
        self.II = self.sparse_matrix_add(self.iiMat, self.H_OO)



        for i in range(len(self.encoder)):
            layer = self.encoder[i]
            if i == 0:  
                userEmbeddings0 = layer(uu_embed0, self.UU, user_index)
                itemEmbeddings0 = layer(ii_embed0, self.II, item_index)
                uiEmbeddings0   = layer(self.ui_embeddings, self.UI, ui_index)
            else:
                userEmbeddings0 = layer(userEmbeddings, self.UU, user_index)
                itemEmbeddings0 = layer(itemEmbeddings, self.II, item_index)
                uiEmbeddings0   = layer(uiEmbeddings,   self.UI, ui_index)
            
            # Aggregation of message features across the two related views in the middle layer then fed into the next layer
            self.ui_userEmbedding0, self.ui_itemEmbedding0 = t.split(uiEmbeddings0, [self.userNum, self.itemNum])
            userEd=( userEmbeddings0 + self.ui_userEmbedding0 )/2.0
            itemEd=( itemEmbeddings0 + self.ui_itemEmbedding0 )/2.0
            userEmbeddings=userEd 
            itemEmbeddings=itemEd
            uiEmbeddings=torch.cat([ userEd,itemEd],0) 
            if norm == 1:
                norm_embeddings = F.normalize(userEmbeddings0, p=2, dim=1)
                self.all_user_embeddings += [norm_embeddings]
                norm_embeddings = F.normalize(itemEmbeddings0, p=2, dim=1)
                self.all_item_embeddings += [norm_embeddings]
                norm_embeddings = F.normalize(uiEmbeddings0, p=2, dim=1)
                self.all_ui_embeddings   += [norm_embeddings]
            else:
                self.all_user_embeddings += [userEmbeddings]
                self.all_item_embeddings += [norm_embeddings]
                self.all_ui_embeddings   += [norm_embeddings]
        self.userEmbedding = t.stack(self.all_user_embeddings, dim=1)
        self.userEmbedding = t.mean(self.userEmbedding, dim = 1)
        self.itemEmbedding = t.stack(self.all_item_embeddings, dim=1)  
        self.itemEmbedding = t.mean(self.itemEmbedding, dim = 1)
        self.uiEmbedding   = t.stack(self.all_ui_embeddings, dim=1)
        self.uiEmbedding   = t.mean(self.uiEmbedding, dim=1)
        self.ui_userEmbedding, self.ui_itemEmbedding = t.split(self.uiEmbedding, [self.userNum, self.itemNum])
        
        # Personalized Transformation of Auxiliary Domain Features
        metatsuembed,metatsiembed = self.metafortansform(self.userEmbedding, self.ui_userEmbedding, self.itemEmbedding, self.ui_itemEmbedding)
        self.userEmbedding = self.userEmbedding + metatsuembed # E_{uu}+E^M_{uu}
        self.itemEmbedding = self.itemEmbedding + metatsiembed

        return self.userEmbedding, self.itemEmbedding,(self.args.wu1*self.ui_userEmbedding + self.args.wu2*self.userEmbedding), (self.args.wi1*self.ui_itemEmbedding + self.args.wi2*self.itemEmbedding), self.ui_userEmbedding, self.ui_itemEmbedding
        

class GCN_layer(nn.Module):
    def __init__(self):
        super(GCN_layer, self).__init__()
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return (d_mat_inv_sqrt).dot(adj).dot(d_mat_inv_sqrt).tocoo()    

    def forward(self, features, Mat, index):
        subset_Mat = Mat
        subset_features = features
        subset_Mat = self.normalize_adj(subset_Mat)
        subset_sparse_tensor = self.sparse_mx_to_torch_sparse_tensor(subset_Mat).cuda()
        out_features = torch.spmm(subset_sparse_tensor, subset_features)
        new_features = torch.empty(features.shape).cuda()
        new_features[index] = out_features
        dif_index = np.setdiff1d(torch.arange(features.shape[0]), index)
        new_features[dif_index] = features[dif_index]
        return new_features

class MLP(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre =   nn.Linear(input_dim, feature_dim,bias=True)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out =    nn.Linear(feature_dim, output_dim,bias=True)

    def forward(self, data):
        x = data
        if self.feature_pre:
            x = self.linear_pre(x)
        prelu=nn.PReLU().cuda()
        x = prelu(x) 
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.tanh(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        return x
















