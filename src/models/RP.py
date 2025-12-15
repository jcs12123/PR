
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph

class RP(GeneralRecommender):
    def __init__(self, config, dataset):
        super(RP, self).__init__(config, dataset)
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.cl_temper=config['cl_temper']
        # 新增对比学习参数
        self.cl_temp = config['cl_temp']  # 温度参数
        self.cl_coef = config['cl_coef']  # 对比损失系数
        self.eps = config['eps']  # 噪声强度
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)
        self.gate_cl_coef = config['gate_cl_coef']
        self.gate_cl_temp = config['gate_cl_temp']
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()
        # ------------------- 新增: 加载并处理用户-用户图 -------------------
        self.n_user_layers = config['uu_layers']  # 用户图 GCN 层数, 默认为 1
        user_graph_file = os.path.join(dataset_path, config['user_graph_dict_file'])

        if os.path.exists(user_graph_file):
            print(f"Loading User-User graph from {user_graph_file}")
            user_graph_dict = np.load(user_graph_file, allow_pickle=True).item()

            rows, cols, weights = [], [], []
            for user_id, (neighbors, edge_weights) in user_graph_dict.items():
                rows.extend([user_id] * len(neighbors))
                cols.extend(neighbors)
                weights.extend(edge_weights)  # 使用从文件加载的权重
            user_adj = sp.coo_matrix((weights, (rows, cols)), shape=(self.n_users, self.n_users), dtype=np.float32)
            user_adj = user_adj + sp.eye(user_adj.shape[0])  # 添加自环 (非常重要)
            rowsum = np.array(user_adj.sum(1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            norm_user_adj = d_mat_inv_sqrt.dot(user_adj).dot(d_mat_inv_sqrt).tocoo()

            self.user_adj = self.sparse_mx_to_torch_sparse_tensor(norm_user_adj).float().to(self.device)
            print("User-User graph loaded and processed successfully.")

        else:
            print(f"Error: User-User graph file not found at {user_graph_file}! U-U GCN will be skipped.")
            self.user_adj = None

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.gate_v_modality = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.gate_v_behavior = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.gate_t_modality = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.gate_t_behavior = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.tau = 0.5
        self.image_augmentation_encoder = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.text_augmentation_encoder = nn.Linear(self.embedding_dim, self.embedding_dim)
        torch.nn.init.eye_(self.image_augmentation_encoder.weight)
        torch.nn.init.zeros_(self.image_augmentation_encoder.bias)
        torch.nn.init.eye_(self.text_augmentation_encoder.weight)
        torch.nn.init.zeros_(self.text_augmentation_encoder.bias)
        self.view_cl_coef = config['cl_coef']

    def pre_epoch_processing(self):
        pass

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            # norm_adj = adj.dot(d_mat_inv)
            # print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, adj, train=False):
        if self.v_feat is not None:
            raw_image_feats = self.image_trs(self.image_embedding.weight)
            item_id_embeds = self.item_id_embedding.weight
            gate_v_signal_from_modality = self.gate_v_modality(raw_image_feats)
            gate_v_signal_from_behavior = self.gate_v_behavior(item_id_embeds)
            gate_v = torch.sigmoid(gate_v_signal_from_modality + gate_v_signal_from_behavior)
            image_item_embeds = torch.multiply(item_id_embeds, gate_v)

        if self.t_feat is not None:
            raw_text_feats = self.text_trs(self.text_embedding.weight)
            gate_t_signal_from_modality = self.gate_t_modality(raw_text_feats)
            gate_t_signal_from_behavior = self.gate_t_behavior(item_id_embeds)
            gate_t = torch.sigmoid(gate_t_signal_from_modality + gate_t_signal_from_behavior)
            text_item_embeds = torch.multiply(item_id_embeds, gate_t)

        # User-Item View
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)

        content_embeds = all_embeddings
        content_user_embeds, content_item_embeds = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        if self.user_adj is not None:
            user_all_layers_embeds = [content_user_embeds]  # 存储每一层的嵌入
            current_user_embeds = content_user_embeds

            for i in range(self.n_user_layers):
                current_user_embeds = torch.sparse.mm(self.user_adj, current_user_embeds)
                user_all_layers_embeds.append(current_user_embeds)

            user_aggregated_embeds = torch.stack(user_all_layers_embeds, dim=1)
            final_content_user_embeds = user_aggregated_embeds.mean(dim=1, keepdim=False)

            content_embeds = torch.cat([final_content_user_embeds, content_item_embeds], dim=0)
        else:
            print("Warning: Skipping U-U GCN propagation as user_adj is None.")
            content_embeds = torch.cat([content_user_embeds, content_item_embeds], dim=0)

        # Item-Item View
        if self.sparse:
            for i in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        else:
            for i in range(self.n_layers):
                image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)
        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        if train:
            self.image_item_embeds_main = image_item_embeds
            noise_image = torch.rand_like(self.image_item_embeds_main).cuda()
            noise_image = F.normalize(noise_image, dim=-1) * self.eps
            self.image_intermediate_rep = self.image_item_embeds_main + noise_image
            self.image_item_embeds_aug = self.image_augmentation_encoder(self.image_intermediate_rep)

            # --- 处理文本模态 ---
            self.text_item_embeds_main = text_item_embeds
            noise_text = torch.rand_like(self.text_item_embeds_main).cuda()
            noise_text = F.normalize(noise_text, dim=-1) * self.eps
            self.text_intermediate_rep = self.text_item_embeds_main + noise_text
            self.text_item_embeds_aug = self.text_augmentation_encoder(self.text_intermediate_rep)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        att_common = torch.cat([self.query_common(image_embeds), self.query_common(text_embeds)], dim=-1)
        weight_common = self.softmax(att_common)
        common_embeds = weight_common[:, 0].unsqueeze(dim=1) * image_embeds + weight_common[:, 1].unsqueeze(
            dim=1) * text_embeds
        sep_image_embeds = image_embeds - common_embeds
        sep_text_embeds = text_embeds - common_embeds
        image_prefer = self.gate_image_prefer(content_embeds)
        text_prefer = self.gate_text_prefer(content_embeds)
        sep_image_embeds = torch.multiply(image_prefer, sep_image_embeds)
        sep_text_embeds = torch.multiply(text_prefer, sep_text_embeds)
        side_embeds = (sep_image_embeds + sep_text_embeds + common_embeds) / 3
        all_embeds = content_embeds + side_embeds
        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(
            self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items],self.cl_temper) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], self.cl_temper)

        cl_loss_image = self.InfoNCE(self.image_item_embeds_main, self.image_item_embeds_aug, self.cl_temp)
        cl_loss_text = self.InfoNCE(self.text_item_embeds_main, self.text_item_embeds_aug, self.cl_temp)
        dmcc_loss = self.view_cl_coef * (cl_loss_image + cl_loss_text)
        total_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss + dmcc_loss
        return total_loss
    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores