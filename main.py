# coding=UTF-8
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dataloader  
import torch.optim as optim
import pickle
import random
import numpy as np
import time
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import os
from ToolScripts.TimeLogger import log
from ToolScripts.BPRData import BPRData  
import ToolScripts.evaluate as evaluate
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model  import MODEL
from args  import make_args


modelUTCStr = str(int(time.time()))
device_gpu = t.device("cuda")

isLoadModel = False



class Hope():
    def __init__(self, args, data, distanceMat, hyperedges):
        self.args = args 
        # (train_sat_sat,orbit_orbit,UiDistance_mat,sat_orbit) = distanceMat
        self.userDistanceMat, self.itemDistanceMat, self.uiDistanceMat,_ = distanceMat
        self.userMat = (self.userDistanceMat != 0) * 1 # train_sat_sat
        self.itemMat = (self.itemDistanceMat != 0) * 1 # orbit_orbit
        self.uiMat = (self.uiDistanceMat != 0) * 1     # UiDistance_mat
       
        self.train_emb_Mat, test_data = data # train_emb_Mat:train_sat_orbit; test_data：test_sat_sat 1pos+99neg 
        self.train_rs_Mat = self.userMat    # train_rs_Mat: train_sat_sat

        self.userNum = self.userMat.shape[0]
        self.itemNum = self.itemMat.shape[0]

        self.H_SS, self.H_OO, self.H_SO = hyperedges

        # 将train_rs_mat 改造成pairs， test的已经在数据处理的时候做了。
        train_coo = self.train_rs_Mat.tocoo()
        train_u, train_v, train_r = train_coo.row, train_coo.col, train_coo.data
        assert np.sum(train_r == 0) == 0
        train_data = np.hstack((train_u.reshape(-1,1),train_v.reshape(-1,1))).tolist() # [train_userNum x 2]

        train_dataset = BPRData(train_data, self.userNum, self.train_rs_Mat, 1, True)
        test_dataset =  BPRData(test_data, self.userNum, self.train_rs_Mat, 0, False)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=self.args.batch, shuffle=True, num_workers=0) 
        self.test_loader = dataloader.DataLoader(test_dataset, batch_size=1024*1000, shuffle=False,num_workers=0)
        self.train_losses = []
        self.test_hr = []
        self.test_ndcg = []
        self.test_pre = []
    
    def prepareModel(self):
        np.random.seed(args.seed)
        t.manual_seed(args.seed)
        t.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        self.model = MODEL(
                           self.args,
                           self.userNum,
                           self.itemNum,
                           self.userMat,self.itemMat, self.uiMat,
                           self.H_SS,self.H_OO,self.H_SO,
                           self.args.hide_dim,
                           self.args.Layers).cuda()
        self.opt = optim.Adam(self.model.parameters(), lr=self.args.lr)

    def predictModel(self,user, pos_i, neg_j, isTest=False):
        if isTest:
            pred_pos = t.sum(user * pos_i, dim=1) #两个向量内积。
            return pred_pos
        else:
            pred_pos = t.sum(user * pos_i, dim=1)
            pred_neg = t.sum(user * neg_j, dim=1)
            return pred_pos, pred_neg

    def adjust_learning_rate(self):
        if self.opt != None:
            for param_group in self.opt.param_groups:
                param_group['lr'] = max(param_group['lr'] * self.args.decay, self.args.minlr)

    def getModelName(self):
        title = "SR-HAN" + "_"
        ModelName = title + self.args.dataset + "_" + modelUTCStr +\
        "_hide_dim_" + str(self.args.hide_dim) +\
        "_lr_" + str(self.args.lr) +\
        "_reg_" + str(self.args.reg) +\
        "_topK_" + str(self.args.topk)+\
        "-ssl_ureg_" + str(self.args.ssl_ureg) +\
        "-ssl_ireg_" + str(self.args.ssl_ireg)
        return ModelName

    def saveHistory(self): 
        history = dict()
        history['loss'] = self.train_losses
        history['hr'] = self.test_hr
        history['ndcg'] = self.test_ndcg
        history['pre'] = self.test_pre
        ModelName = self.getModelName()
        with open(r'./History/' + dataset + r'/' + ModelName + '.his', 'wb') as fs:
            pickle.dump(history, fs)

    def saveModel(self): 
        ModelName = self.getModelName()
        history = dict()
        history['loss'] = self.train_losses
        history['hr'] = self.test_hr
        history['ndcg'] = self.test_ndcg
        history['pre'] = self.test_pre
        savePath = r'./Model/' + dataset + r'/' + ModelName + r'.pth'
        params = {
            'model': self.model,
            'epoch': self.curEpoch,
            'args': self.args,
            'opt': self.opt,
            'history':history
            }
        t.save(params, savePath)
        log("save model : " + ModelName)

    def loadModel(self, modelPath):
        checkpoint = t.load(r'./Model/' + dataset + r'/' + modelPath + r'.pth')
        self.curEpoch = checkpoint['epoch'] + 1
        self.model = checkpoint['model']
        self.args = checkpoint['args']
        self.opt = checkpoint['opt']
        history = checkpoint['history']
        self.train_losses = history['loss']
        self.test_hr = history['hr']
        self.test_ndcg = history['ndcg']
        self.test_pre = history['pre']
        log("load model %s in epoch %d"%(modelPath, checkpoint['epoch']))
    
    # Contrastive Learning
    def ssl_loss(self, data1, data2, index):
        index = t.unique(index)
        embeddings1 = data1[index]
        embeddings2 = data2[index]
        # 对嵌入进行归一化
        norm_embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        norm_embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        
        # 计算正样本对的得分
        # pos_score = t.sum(t.mul(norm_embeddings1, norm_embeddings2), dim=1)
        # pos_score_exp = t.exp(pos_score / self.args.ssl_temp)
        
        # 计算所有样本对的得分
        all_score = t.mm(norm_embeddings1, norm_embeddings2.T)
        all_score_exp = t.exp(all_score / self.args.ssl_temp)
        
        # 计算负样本对的置信度 w_I(i, j)
        cosine_distance = 1 - F.cosine_similarity(norm_embeddings1.unsqueeze(1), norm_embeddings2.unsqueeze(0), dim=2)
        # cosine_distance = 1 - all_score  # 1 - cosine_similarity = cosine_distance
        w_I = 2 * self.args.alpha * t.sigmoid(-self.args.tau_I * cosine_distance)
        w_I[t.arange(len(index)), t.arange(len(index))] = 1.0

        
        # 计算 p_I(i, j)
        p_I = all_score_exp / t.sum(all_score_exp, dim=1, keepdim=True)
        # ssl_loss = -t.sum(w_I * t.log(p_I))/len(index) #对应元素相乘
        # 这样不行主要是neg loss还需要单独平均一下。

        # 第二种方法：计算对比学习的损失函数( neg_loss 分开来算)
        # Calculate loss for each sample
        log_p_I = -t.log(p_I)
        pos_log_p_I = log_p_I.diag()  # Positive pairs' log probability

        # Calculate the weighted sum of negative pairs' log probability
        weighted_neg_log_p_I = w_I * log_p_I
        weighted_neg_log_p_I=weighted_neg_log_p_I.fill_diagonal_(0)  # Ensure positive pairs' weight is 0

        # Calculate the final loss for each sample
        loss_I = pos_log_p_I + t.mean(weighted_neg_log_p_I, dim=1)

        # Calculate the average loss
        ssl_loss = t.sum(loss_I) / len(index)
        
        return ssl_loss
    

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = t.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = t.from_numpy(sparse_mx.data).float()
        shape = t.Size(sparse_mx.shape)
        return t.sparse.FloatTensor(indices, values, shape)
    
    def metaregular(self,em0,em,adj):
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[:,t.randperm(embedding.shape[1])]
            corrupted_embedding = corrupted_embedding[t.randperm(embedding.shape[0])]
            return corrupted_embedding
        def score(x1,x2):
            x1=F.normalize(x1,p=2,dim=-1)
            x2=F.normalize(x2,p=2,dim=-1)
            return t.sum(t.multiply(x1,x2),1)
        user_embeddings = em
        Adj_Norm =t.from_numpy(np.sum(adj,axis=1)).float().cuda()
        adj=self.sparse_mx_to_torch_sparse_tensor(adj)
        edge_embeddings = t.spmm(adj.cuda(),user_embeddings)/Adj_Norm
        user_embeddings=em0
        graph = t.mean(edge_embeddings,0)
        pos   = score(user_embeddings,graph)
        neg1  = score(row_column_shuffle(user_embeddings),graph)
        global_loss = t.mean(-t.log(t.sigmoid(pos-neg1)))
        return global_loss 
    
    # Model train
    def trainModel(self):
        epoch_loss = 0
        self.train_loader.dataset.ng_sample() 
        step_num = 0 # count batch num
        for user, item_i, item_j in self.train_loader:  
            user = user.long().cuda()
            item_i = item_i.long().cuda()
            item_j = item_j.long().cuda()  
            step_num += 1
            self.train= True
            itemindex = t.unique(t.cat((item_i, item_j)))
            userindex = t.unique(user)
            self.userEmbed, self.itemEmbed, self.ui_userEmbedall, self.ui_itemEmbedall, self.ui_userEmbed, self.ui_itemEmbed = self.model( self.train, norm=1)
            # E_{uu}+E^M_{uu} ,_, E^F_u,E^F_i, E_u, E_i, meta regloss.


            # Contrastive Learning of collaborative relations
            ssl_loss_user = self.ssl_loss(self.ui_userEmbed, self.userEmbed, userindex)    
            ssl_loss_item = self.ssl_loss(self.ui_userEmbed, self.userEmbed, itemindex)
            ssl_loss = self.args.ssl_ureg * ssl_loss_user +  self.args.ssl_ireg * ssl_loss_item
            
            # prediction
            pred_pos, pred_neg = self.predictModel(self.ui_userEmbedall[user],  self.ui_userEmbedall[item_i],  self.ui_userEmbedall[item_j]) 

            bpr_loss = - nn.LogSigmoid()(pred_pos - pred_neg).sum()  
            epoch_loss += bpr_loss.item()
            regLoss = (t.norm(self.ui_userEmbedall[user])**2 + t.norm( self.ui_userEmbedall[item_i])**2 + t.norm( self.ui_userEmbedall[item_j])**2) 

            # Regularization: the constraint of transformed reasonableness
            metaregloss = 0

            self.reg_lossu = self.metaregular((self.ui_userEmbed[userindex.cpu().numpy()]),(self.userEmbed),self.userMat[userindex.cpu().numpy()])
            self.reg_lossi = self.metaregular((self.ui_userEmbed[itemindex.cpu().numpy()]),(self.userEmbed),self.userMat[itemindex.cpu().numpy()])
            metaregloss =  (self.reg_lossu +  self.reg_lossi)/2.0


            loss = ((bpr_loss + regLoss * self.args.reg ) / self.args.batch) + ssl_loss*self.args.ssl_beta + metaregloss*self.args.metareg
            
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),  max_norm=20, norm_type=2)
            self.opt.step()
        return epoch_loss 

    def testModel(self):
        HR=[]
        NDCG=[]
        PRE=[]
        with t.no_grad():
            uid = np.arange(0,self.userNum)
            iid = np.arange(0,self.itemNum)
            self.train = False
            self.userEmbed, self.itemEmbed, self.ui_userEmbed, self.ui_itemEmbed,_,_= self.model( self.train,norm=1)
            for test_u, test_i in self.test_loader:
                test_u = test_u.long().cuda()
                test_i = test_i.long().cuda()
                # pred = self.predictModel( self.ui_userEmbed[test_u], self.ui_itemEmbed[test_i], None, isTest=True)
                pred = self.predictModel( self.ui_userEmbed[test_u], self.ui_userEmbed[test_i], None, isTest=True)

                batch = int(test_u.cpu().numpy().size/100)
                for i in range(batch):
                    batch_socres=pred[i*100:(i+1)*100].view(-1)
                    _,indices=t.topk(batch_socres,self.args.topk) 
                    tmp_item_i=test_i[i*100:(i+1)*100]
                    recommends=t.take(tmp_item_i,indices).cpu().numpy().tolist() # k
                    gt_item=tmp_item_i[0].item() # 1

                    HR.append(evaluate.hit(gt_item,recommends))
                    NDCG.append(evaluate.ndcg(gt_item,recommends))

                    # 在前 top_k 个推荐项目中，包含 ground truth 项目的比例。
                    # precision_at_k = len([rec for rec in recommends if rec == gt_item]) / self.args.topk
                    # PRE.append(precision_at_k)
                    PRE.append([len([rec for rec in recommends[:k] if rec == gt_item])/ k for k in [1, 2, 3, 4, 5, 10]])


        return np.mean(HR),np.mean(NDCG),np.array(PRE).mean(axis=0).tolist()
    
    def run(self):
        self.prepareModel()
        if isLoadModel:
            # self.loadModel(LOAD_MODEL_PATH)
            HR,NDCG,PRE = self.testModel()
            # log("HR@10=%.4f, NDCG@10=%.4f, PRE@10=%.4f"%(HR, NDCG,PRE))
            log(" HR@10=%.4f, NDCG@10=%.4f, PRE@1=%.4f, PRE@2=%.4f, PRE@3=%.4f, PRE@4=%.4f, PRE@5=%.4f, PRE@10=%.4f" %(HR, NDCG, PRE[0], PRE[1], PRE[2], PRE[3], PRE[4], PRE[5]))
            return 
        self.curEpoch = 0
        best_hr=-1
        best_ndcg=-1
        best_epoch=-1
        best_pre =-1
        HR_lis=[]
        wait=0
        for e in range(args.epochs+1):
            self.curEpoch = e
            # train
            log("**************************************************************")
            epoch_loss  = self.trainModel()
            self.train_losses.append(epoch_loss)
            log("epoch %d/%d, epoch_loss=%.2f"%(e, args.epochs, epoch_loss))

            # test
            HR,NDCG,PRE = self.testModel()
            self.test_hr.append(HR)
            self.test_ndcg.append(NDCG)
            self.test_pre.append(PRE)
            # log("epoch %d/%d, HR@10=%.4f, NDCG@10=%.4f, PRE@10=%.4f"%(e, args.epochs, HR, NDCG,PRE))
            log("epoch %d/%d, HR@10=%.4f, NDCG@10=%.4f, PRE@1=%.4f, PRE@2=%.4f, PRE@3=%.4f, PRE@4=%.4f, PRE@5=%.4f, PRE@10=%.4f" %(e, args.epochs, HR, NDCG, PRE[0], PRE[1], PRE[2], PRE[3], PRE[4], PRE[5]))

            self.adjust_learning_rate()     
            if HR>best_hr:
                best_hr,best_ndcg,best_pre,best_epoch = HR,NDCG,PRE,e
                wait=0
                # self.saveModel()
            else:
                wait+=1
                print('wait=%d'%(wait))
            HR_lis.append(HR)
            self.saveHistory()
            if wait==self.args.patience:
                log('Early stop! best epoch = %d'%(best_epoch))
                # self.loadModel(self.getModelName())
                break

        print("*****************************")
        log("best epoch %d/%d, HR@10=%.4f, NDCG@10=%.4f, PRE@1=%.4f, PRE@2=%.4f, PRE@3=%.4f, PRE@4=%.4f, PRE@5=%.4f, PRE@10=%.4f" %(best_epoch,best_hr,best_ndcg, best_pre[0], best_pre[1], best_pre[2], best_pre[3], best_pre[4], best_pre[5]))

        # log("best epoch = %d, HR= %.4f, NDCG=%.4f, PRE@10=%.4f"% (best_epoch,best_hr,best_ndcg,best_pre)) 
        print("*****************************")   
        print(self.args)
        self.saveModel()
        log("model name : %s"%(self.getModelName()))
       
if __name__ == '__main__':
    # hyper parameters
    args = make_args()
    print(args)
    dataset = args.dataset

    # train & test data
    with open(r'dataset/'+args.dataset+'/data.pkl', 'rb') as fs:
        data = pickle.load(fs)
    with open(r'dataset/'+ args.dataset + '/distanceMat.pkl', 'rb') as fs:
        distanceMat = pickle.load(fs) 
    with open(r'dataset/'+ args.dataset + '/hyperedges.pkl', 'rb') as fs:
        hyperedges = pickle.load(fs) 


    # model instance
    hope = Hope(args, data, distanceMat, hyperedges)
    modelName = hope.getModelName()
    print('ModelName = ' + modelName)    
    hope.run()
   

    

  

