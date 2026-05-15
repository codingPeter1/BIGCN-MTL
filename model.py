import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from data_set import DataSet
from utils import BPRLoss, EmbLoss, his_Loss, MFLogLoss,BCR_loss


class BIGCN_MTL(nn.Module):
    def __init__(self, args, dataset: DataSet, LastStage_embeddings, embeddings_load_path):
        super(BIGCN_MTL, self).__init__()

        self.dateset = dataset
        self.LastStage_embeddings = LastStage_embeddings

        self.embeddings_load_path = embeddings_load_path
        self.device = args.device
        self.layers = args.layers
        self.message_dropout = nn.Dropout(p=args.message_dropout)
        self.node_dropout = nn.Dropout(p=args.node_dropout)
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.edge_index = dataset.edge_index
        self.behaviors_adj = dataset.behaviors_adj
        self.behaviors = args.behaviors
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.behavior_layer_emb = {b: [] for b in self.behaviors}
        self.behaviors_layers = {behavior: args.layers[index] for index, behavior in enumerate(self.behaviors)}


        self.reg_weight = args.reg_weight
        self.his_weight = args.his_weight
        self.kd_weight = args.kd_weight
        self.inner_weight = args.inner_weight
        self.tao = args.tao

        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        self.his_loss = his_Loss()
        self.MFlog_loss = MFLogLoss()
        self.BCR = BCR_loss()

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model

        self.storage_all_embeddings = None
        self.Denominator = nn.Sequential(nn.Linear(1, 1, bias=False), nn.ReLU(inplace=False))
        self.old_scale = nn.Sequential(nn.Linear(1, 1, bias=False), nn.ReLU(inplace=False))
        self.conv = {behavior: nn.ModuleList(
            [nn.Conv2d(1, 1, (2, 1), stride=1, bias=False).to(self.device) for _ in
             range(self.behaviors_layers[behavior])]) for
            behavior in self.behaviors}


        self.apply(self._init_weights)
        self._load_model()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if isinstance(module, nn.MultiheadAttention):
            pass 

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(self.embeddings_load_path, map_location=self.device)
            self.load_state_dict(parameters, strict=False)
            print('============================load model successfully=================================')
            logger.info(f"load model from {self.embeddings_load_path}")

    def denominator_forward(self, Degree_new, Degree_old):
        x_Denominator = Degree_old + Degree_new
        x_molecular = torch.ones_like(x_Denominator)
        return x_molecular, x_Denominator

    def oldscale_forward(self, old_scale):
        if torch.isnan(old_scale).any() or torch.isinf(old_scale).any():
            raise ValueError("Input to oldscale_forward contains NaN or Inf")

        old_scale = self.old_scale(old_scale)

        if torch.isnan(old_scale).any() or torch.isinf(old_scale).any():
            raise ValueError("Output of oldscale_forward contains NaN or Inf")
        return old_scale

    def transfer_forward(self, x_old, x_new, behavior, layer):
        x = torch.cat((x_old, x_new), dim=-1)
        x = x.view(-1, 1, 2, x_new.shape[-1])
        x = self.conv[behavior][layer](x)
        x = x.view(-1, x_new.shape[-1])
        return x

    def gated_fusion(self, x_old, x_new, behavior, layer):
        concat_emb = torch.cat([x_old, x_new], dim=-1)

        gate = torch.sigmoid(self.gate_net[behavior][layer](concat_emb))

        return gate * x_old + (1 - gate) * x_new

    def get_layers_weight(self, behavior, all_emb):
        embs = [all_emb]
        allembs_list = [all_emb]

        graph = self.behaviors_adj[behavior].to(self.device)
        now_user_degree, now_item_degree, old_user_degree, old_item_degree = self.dateset.get_degree(behavior)
        degree_molecular, degree_Denominator = self.denominator_forward(
            torch.cat((now_user_degree, now_item_degree), dim=0), torch.cat((old_user_degree, old_item_degree), dim=0))

        degree_Denominator = degree_Denominator.pow(0.5)
        norm_degree = torch.div(degree_molecular, (degree_Denominator + 1e-9))
        norm_degree = norm_degree.flatten()
        for layer in range(self.behaviors_layers[behavior]):
            all_emb = torch.mul(norm_degree.view(-1, 1), all_emb)
            all_emb = torch.sparse.mm(graph, all_emb)
            all_emb = torch.mul(norm_degree.view(-1, 1), all_emb)
            all_emb = self.node_dropout(all_emb)
            embs.append(all_emb)
            allembs_list.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users + 1, self.n_items + 1])
        return light_out, users, items, allembs_list, degree_molecular, degree_Denominator, old_user_degree, old_item_degree

    def contrastive_loss(self, emb1, emb2, temperature=0.1):
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)

        sim_matrix = torch.matmul(emb1, emb2.T) / temperature

        labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def feature_level_transfer(self, current_emb, history_emb, behavior):
        
        concat_emb = torch.cat([current_emb, history_emb], dim=-1)
        transfer_gate = torch.sigmoid(self.feature_transfer[behavior](concat_emb))

        transferred_emb = current_emb + transfer_gate * (history_emb - current_emb)
        return transferred_emb

    def gcn_propagate(self):
        all_embeddings = {}
        total_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        for behavior in self.behaviors:
            layer_embeddings = total_embeddings
            mean_embeddings, _0, _1, allLayerEmbs, degree_molecular, degree_Denominator, old_user_degree, old_item_degree = self.get_layers_weight(
                behavior, layer_embeddings)
            del _0, _1
            old_degree = torch.cat([old_user_degree, old_item_degree], dim=0)
            old_scale = old_degree.pow(0.5)
            old_scale = torch.mul(degree_molecular, old_scale)
            new_scale = degree_Denominator
            rscale_vec = torch.div(old_scale, new_scale + 1e-9)
            old_layer_embeddings = self.LastStage_embeddings[behavior]
            new_layer_embeddings = [total_embeddings]
            for layer in range(self.behaviors_layers[behavior]):
                old_layer_embeddings[layer] = old_layer_embeddings[layer].to(self.device)
                t_layer = old_layer_embeddings[layer] * rscale_vec
                new_layer_embeddings.append(
                    self.transfer_forward(t_layer, allLayerEmbs[layer], behavior, layer))

            layer_embeddings = F.normalize(new_layer_embeddings[-1], dim=-1)
            total_embeddings = layer_embeddings + total_embeddings

            all_embeddings[behavior] = total_embeddings

        return all_embeddings

    def forward(self, batch_data):
        self.storage_all_embeddings = None
        initial_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = self.gcn_propagate()
        total_loss = 0.0
        his_loss = 0.0
        kd_loss = 0.0

        for index, behavior in enumerate(self.behaviors):
            data = batch_data[:, index]
            users = data[:, 0].long()
            items = data[:, 1:].long()
            user_all_embedding, item_all_embedding = torch.split(all_embeddings[behavior],
                                                                 [self.n_users + 1, self.n_items + 1])
            if index==0:
                his_user_embedding, item_his_embedding = torch.split(initial_emb+F.normalize(self.LastStage_embeddings[self.behaviors[index]][-1], dim=-1),
                                                                 [self.n_users + 1, self.n_items + 1])
            else:
                his_user_embedding, item_his_embedding = torch.split(self.LastStage_embeddings[self.behaviors[index-1]][-1]+F.normalize(self.LastStage_embeddings[self.behaviors[index]][-1], dim=-1),
                                                                 [self.n_users + 1, self.n_items + 1])

            user_feature = user_all_embedding[users.view(-1, 1)].expand(-1, items.shape[1], -1)
            item_feature = item_all_embedding[items]
            user_feature, item_feature = self.message_dropout(user_feature), self.message_dropout(item_feature)
            scores = torch.sum(user_feature * item_feature, dim=2)
            total_loss += self.bpr_loss(scores[:, 0], scores[:, 1])

            users = batch_data[:, 0, 0].long()
            current_user_emb = user_all_embedding[users]
            history_user_emb = his_user_embedding[users]

            his_loss += self.contrastive_loss(current_user_emb, history_user_emb,self.tao)

            kd_loss_items = items[:, 1].long()
            kd_now_user_feature = user_all_embedding[users]
            kd_his_user_feature = his_user_embedding[users]
            kd_now_item_feature = item_all_embedding[kd_loss_items]
            kd_his_item_feature = item_his_embedding[kd_loss_items]

            kd_scores1 = torch.sum(kd_now_user_feature * kd_now_item_feature, dim=1)
            kd_scores2 = torch.sum( kd_his_user_feature * kd_his_item_feature, dim=1)
            kd_loss_item = (kd_scores1-kd_scores2)**2
            kd_loss +=  kd_loss_item.mean()

        total_loss = (total_loss + self.reg_weight * self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)+ self.his_weight * his_loss + self.kd_weight* kd_loss)
        return total_loss

    def full_predict(self, users):
        if self.storage_all_embeddings is None:
            self.storage_all_embeddings = self.gcn_propagate()

        user_embedding, item_embedding = torch.split(self.storage_all_embeddings[self.behaviors[-1]],
                                                     [self.n_users + 1, self.n_items + 1])
        user_emb = user_embedding[users.long()]
        scores = torch.matmul(user_emb, item_embedding.transpose(0, 1))
        return scores

    def get_saved_layer_embs(self):
        all_embeddings = {}
        total_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        for behavior in self.behaviors:
            layer_embeddings = total_embeddings
            mean_embeddings, _0, _1, allLayerEmbs, degree_molecular, degree_Denominator, old_user_degree, old_item_degree = self.get_layers_weight(
                behavior, layer_embeddings)
            del _0, _1
            old_degree = torch.cat([old_user_degree, old_item_degree], dim=0)
            old_scale = self.oldscale_forward(old_degree)
            old_scale = old_scale.pow(0.5)
            old_scale = torch.mul(degree_molecular, old_scale)
            new_scale = degree_Denominator
            rscale_vec = torch.div(old_scale, new_scale + 1e-9)
            old_layer_embeddings = self.LastStage_embeddings[behavior]
            new_layer_embeddings = [total_embeddings]
            for layer in range(self.behaviors_layers[behavior]):
                old_layer_embeddings[layer] = old_layer_embeddings[layer].to(self.device)
                layer_temp = old_layer_embeddings[layer] * rscale_vec
                new_layer_embeddings.append(
                    self.transfer_forward(layer_temp, allLayerEmbs[layer], behavior, layer))
            self.behavior_layer_emb[behavior] = new_layer_embeddings[1:]

            layer_embeddings = F.normalize(new_layer_embeddings[-1], dim=-1)
            total_embeddings = layer_embeddings + total_embeddings
        return self.behavior_layer_emb