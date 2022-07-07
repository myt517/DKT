from util import *
import torch.nn as nn
import torch
from torch.nn.functional import normalize
import math
from keras.utils.np_utils import to_categorical
from contrastive_loss import *

def onehot_labelling(int_labels, num_classes):
    categorical_labels = to_categorical(int_labels, num_classes=num_classes)
    return categorical_labels

def pair_cosine_similarity(x, x_adv, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    n_adv = x_adv.norm(p=2, dim=1, keepdim=True)
    #print(x.shape)
    #print(x_adv.shape)
    #print(n.shape)
    #print(n_adv.shape)
    #print((n * n.t()).shape)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps), (x_adv @ x_adv.t()) / (n_adv * n_adv.t()).clamp(min=eps), (x @ x_adv.t()) / (n * n_adv.t()).clamp(min=eps)

def nt_xent(x, x_adv, mask, cuda=True, t=0.5):
    x, x_adv, x_c = pair_cosine_similarity(x, x_adv)
    x = torch.exp(x / t)
    x_adv = torch.exp(x_adv / t)
    x_c = torch.exp(x_c / t)
    mask_count = mask.sum(1)
    mask_reverse = (~(mask.bool())).long()

    if cuda:
        dis = (x * (mask - torch.eye(x.size(0)).long().cuda()) + x_c * mask) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + mask_reverse
        dis_adv = (x_adv * (mask - torch.eye(x.size(0)).long().cuda()) + x_c.T * mask) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t))) + mask_reverse
    else:
        dis = (x * (mask - torch.eye(x.size(0)).long()) + x_c * mask) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + mask_reverse
        dis_adv = (x_adv * (mask - torch.eye(x.size(0)).long()) + x_c.T * mask) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t))) + mask_reverse

    loss = (torch.log(dis).sum(1) + torch.log(dis_adv).sum(1)) / mask_count
    #loss = dis.sum(1) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + dis_adv.sum(1) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t)))
    return -loss.mean()
    #return -torch.log(loss).mean()



class BertForModel(BertPreTrainedModel):
    def __init__(self,config, num_labels):
        super(BertForModel, self).__init__(config)

        self.num_labels = num_labels

        self.bert = BertModel(config) # 这个是backbone
        self.rnn = nn.GRU(input_size=config.hidden_size, hidden_size=config.hidden_size, num_layers=1,
                          dropout=config.hidden_dropout_prob, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # 以上为编码器pooling层
        self.instance_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 128),
        ) # instance-level 投影

        self.cluster_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.num_labels),
        ) # class(cluster)-level 投影

        self.softmax = nn.Softmax(dim=1)

        self.apply(self.init_bert_weights)


    def forward(self, batch1 = None, mode = None, pretrain = True, positive_sample=None, negative_sample=None):
        if pretrain:
            if mode == "pre-trained":
                input_ids, input_mask, segment_ids, label_ids = batch1
                encoded_layer_12, pooled_output = self.bert(input_ids, segment_ids, input_mask,
                                                            output_all_encoded_layers=False)
                _, pooled_output = self.rnn(encoded_layer_12)
                pooled_output = torch.cat((pooled_output[0].squeeze(0),pooled_output[1].squeeze(0)),dim=1)
                pooled_output = self.dense(pooled_output)
                pooled_output = self.activation(pooled_output)
                pooled_output = self.dropout(pooled_output)

                # Class-level 损失函数
                logits = self.cluster_projector(pooled_output)
                ce_loss = nn.CrossEntropyLoss()(logits, label_ids)


                # Instance-level 损失函数
                z_i = self.instance_projector(pooled_output)

                label_ids = label_ids.cpu()
                labels = onehot_labelling(label_ids, self.num_labels)
                labels = torch.from_numpy(labels)
                labels = labels.cuda()
                label_mask = torch.mm(labels, labels.T).bool().long()
                encoded_layer_12_02, pooled_output_02 = self.bert(input_ids, segment_ids, input_mask,
                                                            output_all_encoded_layers=False)
                _, pooled_output_02 = self.rnn(encoded_layer_12_02)
                pooled_output_02 = torch.cat((pooled_output_02[0].squeeze(0), pooled_output_02[1].squeeze(0)), dim=1)
                pooled_output_02 = self.dense(pooled_output_02)
                pooled_output_02 = self.activation(pooled_output_02)
                pooled_output_02 = self.dropout(pooled_output_02)
                z_j = self.instance_projector(pooled_output_02)
                sup_cont_loss = nt_xent(z_i, z_j, label_mask, cuda=True)

                loss = ce_loss + sup_cont_loss

                return loss

            elif mode == "feature-extract":
                input_ids, input_mask, segment_ids, label_ids = batch1
                encoded_layer_12, pooled_output = self.bert(input_ids, segment_ids, input_mask,
                                                            output_all_encoded_layers=False)
                _, pooled_output = self.rnn(encoded_layer_12)
                pooled_output = torch.cat((pooled_output[0].squeeze(0), pooled_output[1].squeeze(0)), dim=1)
                pooled_output = self.dense(pooled_output)
                pooled_output = self.activation(pooled_output)
                pooled_output = self.dropout(pooled_output)

                return pooled_output

            elif mode == "contrastive-clustering":
                input_ids_1, input_mask_1, segment_ids_1, label_ids_1 = batch1

                encoded_layer_12_emb01, pooled_output_01 = self.bert(input_ids_1, segment_ids_1, input_mask_1,
                                                                     output_all_encoded_layers=False)
                encoded_layer_12_emb02, pooled_output_02 = self.bert(input_ids_1, segment_ids_1, input_mask_1,
                                                                     output_all_encoded_layers=False)

                _, pooled_output_01 = self.rnn(encoded_layer_12_emb01)
                _, pooled_output_02 = self.rnn(encoded_layer_12_emb02)

                pooled_output_01 = torch.cat((pooled_output_01[0].squeeze(0), pooled_output_01[1].squeeze(0)), dim=1)
                pooled_output_02 = torch.cat((pooled_output_02[0].squeeze(0), pooled_output_02[1].squeeze(0)), dim=1)

                pooled_output_01 = self.dense(pooled_output_01)
                pooled_output_02 = self.dense(pooled_output_02)

                pooled_output_01 = self.activation(pooled_output_01)
                pooled_output_02 = self.activation(pooled_output_02)

                pooled_output_01 = self.dropout(pooled_output_01)
                pooled_output_02 = self.dropout(pooled_output_02)

                z_i = normalize(self.instance_projector(pooled_output_01), dim=1)
                z_j = normalize(self.instance_projector(pooled_output_02), dim=1)

                c_i = self.cluster_projector(pooled_output_01)
                c_j = self.cluster_projector(pooled_output_02)

                #c_i = self.classifier(c_i)
                #c_j = self.classifier(c_j)

                c_i = self.softmax(c_i)
                c_j = self.softmax(c_j)

                return z_i, z_j, c_i, c_j

            else:
                input_ids, input_mask, segment_ids, label_ids = batch1
                encoded_layer_12, pooled_output = self.bert(input_ids, segment_ids, input_mask,
                                                            output_all_encoded_layers=False)
                _, pooled_output = self.rnn(encoded_layer_12)
                pooled_output = torch.cat((pooled_output[0].squeeze(0), pooled_output[1].squeeze(0)), dim=1)
                pooled_output = self.dense(pooled_output)
                pooled_output = self.activation(pooled_output)
                pooled_output = self.dropout(pooled_output)

                logits = self.cluster_projector(pooled_output)

                feats = normalize(pooled_output, dim=1)

                return feats, logits

        else:
            if mode == "feature-extract":
                input_ids, input_mask, segment_ids, label_ids = batch1
                encoded_layer_12, pooled_output = self.bert(input_ids, segment_ids, input_mask,
                                                            output_all_encoded_layers=False)
                pooled_output = encoded_layer_12.mean(dim=1)
                #pooled_output = self.dense(pooled_output)
                #pooled_output = self.activation(pooled_output)
                #pooled_output = self.dropout(pooled_output)

                return pooled_output

            elif mode == "contrastive-clustering":
                input_ids_1, input_mask_1, segment_ids_1, label_ids_1 = batch1

                encoded_layer_12_emb01, pooled_output_01 = self.bert(input_ids_1, segment_ids_1, input_mask_1,
                                                                     output_all_encoded_layers=False)
                encoded_layer_12_emb02, pooled_output_02 = self.bert(input_ids_1, segment_ids_1, input_mask_1,
                                                                     output_all_encoded_layers=False)

                _, pooled_output_01 = self.rnn(encoded_layer_12_emb01)
                _, pooled_output_02 = self.rnn(encoded_layer_12_emb02)

                pooled_output_01 = torch.cat((pooled_output_01[0].squeeze(0), pooled_output_01[1].squeeze(0)), dim=1)
                pooled_output_02 = torch.cat((pooled_output_02[0].squeeze(0), pooled_output_02[1].squeeze(0)), dim=1)

                pooled_output_01 = self.dense(pooled_output_01)
                pooled_output_02 = self.dense(pooled_output_02)

                #pooled_output_01 = encoded_layer_12_emb01.mean(dim=1)
                #pooled_output_02 = encoded_layer_12_emb02.mean(dim=1)

                #pooled_output_01 = self.dense(pooled_output_01)
                #pooled_output_02 = self.dense(pooled_output_02)

                pooled_output_01 = self.activation(pooled_output_01)
                pooled_output_02 = self.activation(pooled_output_02)

                pooled_output_01 = self.dropout(pooled_output_01)
                pooled_output_02 = self.dropout(pooled_output_02)

                z_i = normalize(self.instance_projector(pooled_output_01), dim=1)
                z_j = normalize(self.instance_projector(pooled_output_02), dim=1)

                c_i = self.cluster_projector(pooled_output_01)
                c_j = self.cluster_projector(pooled_output_02)

                c_i = self.softmax(c_i)
                c_j = self.softmax(c_j)

                return z_i, z_j, c_i, c_j

            else:
                exit()

    def forward_cluster(self, batch, pretrain = True):
        if pretrain:
            input_ids, input_mask, segment_ids, label_ids = batch
            encoded_layer_12, pooled_output = self.bert(input_ids, segment_ids, input_mask,
                                                        output_all_encoded_layers=False)
            _, pooled_output = self.rnn(encoded_layer_12)
            pooled_output = torch.cat((pooled_output[0].squeeze(0), pooled_output[1].squeeze(0)), dim=1)
            pooled_output = self.dense(pooled_output)
            pooled_output = self.activation(pooled_output)
            pooled_output = self.dropout(pooled_output)

            #z = normalize(self.instance_projector(pooled_output), dim=1)
            c = self.cluster_projector(pooled_output)
            #c = self.classifier(feats)
            c = self.softmax(c)

            return c, pooled_output

        else:
            input_ids, input_mask, segment_ids, label_ids = batch
            encoded_layer_12, pooled_output = self.bert(input_ids, segment_ids, input_mask,
                                                        output_all_encoded_layers=False)
            #pooled_output = encoded_layer_12.mean(dim=1)
            #pooled_output = self.dense(pooled_output)
            _, pooled_output = self.rnn(encoded_layer_12)
            pooled_output = torch.cat((pooled_output[0].squeeze(0), pooled_output[1].squeeze(0)), dim=1)
            pooled_output = self.dense(pooled_output)
            pooled_output = self.activation(pooled_output)
            pooled_output = self.dropout(pooled_output)

            #z = normalize(self.instance_projector(pooled_output), dim=1)
            c = self.cluster_projector(pooled_output)
            c = self.softmax(c)

            return c, pooled_output



class BertForModel_kt(BertPreTrainedModel):
    def __init__(self,config, num_labels):
        super(BertForModel_kt, self).__init__(config)
        self.num_labels = num_labels

        self.bert = BertModel(config) # 这个是backbone
        self.rnn = nn.GRU(input_size=config.hidden_size, hidden_size=config.hidden_size, num_layers=1,
                          dropout=config.hidden_dropout_prob, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # 以上为编码器pooling层

        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        '''
        self.instance_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 128),
        ) # instance-level 投影

        self.cluster_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.num_labels),
        ) # class(cluster)-level 投影

        self.softmax = nn.Softmax(dim=1)
        '''
        self.softmax = nn.Softmax(dim=1)
        self.apply(self.init_bert_weights)


    def forward(self, batch1 = None, mode = None, pretrain = True):
        if pretrain:
            if mode == "pre-trained":
                input_ids, input_mask, segment_ids, label_ids = batch1
                encoded_layer_12, pooled_output = self.bert(input_ids, segment_ids, input_mask,
                                                            output_all_encoded_layers=False)
                _, pooled_output = self.rnn(encoded_layer_12)
                pooled_output = torch.cat((pooled_output[0].squeeze(0),pooled_output[1].squeeze(0)),dim=1)
                pooled_output = self.dense(pooled_output)
                pooled_output = self.activation(pooled_output)
                pooled_output = self.dropout(pooled_output)

                # 交叉熵损失函数
                logits = self.classifier(pooled_output)
                ce_loss = nn.CrossEntropyLoss()(logits, label_ids)

                # 监督对比学习损失
                label_ids = label_ids.cpu()
                labels = onehot_labelling(label_ids, self.num_labels)
                labels = torch.from_numpy(labels)
                labels = labels.cuda()
                label_mask = torch.mm(labels, labels.T).bool().long()
                encoded_layer_12_02, pooled_output_02 = self.bert(input_ids, segment_ids, input_mask,
                                                                  output_all_encoded_layers=False)
                _, pooled_output_02 = self.rnn(encoded_layer_12_02)
                pooled_output_02 = torch.cat((pooled_output_02[0].squeeze(0), pooled_output_02[1].squeeze(0)), dim=1)
                pooled_output_02 = self.dense(pooled_output_02)
                pooled_output_02 = self.activation(pooled_output_02)
                pooled_output_02 = self.dropout(pooled_output_02)
                sup_cont_loss = nt_xent(pooled_output, pooled_output_02, label_mask, cuda=True)

                loss = ce_loss + sup_cont_loss

                return loss

            elif mode == "feature-extract":
                input_ids, input_mask, segment_ids, label_ids = batch1
                encoded_layer_12, pooled_output = self.bert(input_ids, segment_ids, input_mask,
                                                            output_all_encoded_layers=False)
                _, pooled_output = self.rnn(encoded_layer_12)
                pooled_output = torch.cat((pooled_output[0].squeeze(0), pooled_output[1].squeeze(0)), dim=1)
                pooled_output = self.dense(pooled_output)
                pooled_output = self.activation(pooled_output)
                pooled_output = self.dropout(pooled_output)

                return pooled_output

            elif mode == "contrastive-clustering":
                input_ids_1, input_mask_1, segment_ids_1, label_ids_1 = batch1

                encoded_layer_12_emb01, pooled_output_01 = self.bert(input_ids_1, segment_ids_1, input_mask_1,
                                                                     output_all_encoded_layers=False)
                encoded_layer_12_emb02, pooled_output_02 = self.bert(input_ids_1, segment_ids_1, input_mask_1,
                                                                     output_all_encoded_layers=False)

                _, pooled_output_01 = self.rnn(encoded_layer_12_emb01)
                _, pooled_output_02 = self.rnn(encoded_layer_12_emb02)

                pooled_output_01 = torch.cat((pooled_output_01[0].squeeze(0), pooled_output_01[1].squeeze(0)), dim=1)
                pooled_output_02 = torch.cat((pooled_output_02[0].squeeze(0), pooled_output_02[1].squeeze(0)), dim=1)

                pooled_output_01 = self.dense(pooled_output_01)
                pooled_output_02 = self.dense(pooled_output_02)

                pooled_output_01 = self.activation(pooled_output_01)
                pooled_output_02 = self.activation(pooled_output_02)

                pooled_output_01 = self.dropout(pooled_output_01)
                pooled_output_02 = self.dropout(pooled_output_02)

                z_i = normalize(pooled_output_01)
                z_j = normalize(pooled_output_02)

                c_i = self.classifier(pooled_output_01)
                c_j = self.classifier(pooled_output_02)

                #c_i = self.classifier(c_i)
                #c_j = self.classifier(c_j)

                c_i = self.softmax(c_i)
                c_j = self.softmax(c_j)

                return z_i, z_j, c_i, c_j

            else:
                input_ids, input_mask, segment_ids, label_ids = batch1
                encoded_layer_12, pooled_output = self.bert(input_ids, segment_ids, input_mask,
                                                            output_all_encoded_layers=False)
                _, pooled_output = self.rnn(encoded_layer_12)
                pooled_output = torch.cat((pooled_output[0].squeeze(0), pooled_output[1].squeeze(0)), dim=1)
                pooled_output = self.dense(pooled_output)
                pooled_output = self.activation(pooled_output)
                pooled_output = self.dropout(pooled_output)

                logits = self.classifier(pooled_output)

                return pooled_output, logits

        else:
            if mode == "feature-extract":
                input_ids, input_mask, segment_ids, label_ids = batch1
                encoded_layer_12, pooled_output = self.bert(input_ids, segment_ids, input_mask,
                                                            output_all_encoded_layers=False)
                pooled_output = encoded_layer_12.mean(dim=1)
                #pooled_output = self.dense(pooled_output)
                #pooled_output = self.activation(pooled_output)
                #pooled_output = self.dropout(pooled_output)

                return pooled_output

            elif mode == "contrastive-clustering":
                input_ids_1, input_mask_1, segment_ids_1, label_ids_1 = batch1

                encoded_layer_12_emb01, pooled_output_01 = self.bert(input_ids_1, segment_ids_1, input_mask_1,
                                                                     output_all_encoded_layers=False)
                encoded_layer_12_emb02, pooled_output_02 = self.bert(input_ids_1, segment_ids_1, input_mask_1,
                                                                     output_all_encoded_layers=False)

                _, pooled_output_01 = self.rnn(encoded_layer_12_emb01)
                _, pooled_output_02 = self.rnn(encoded_layer_12_emb02)

                pooled_output_01 = torch.cat((pooled_output_01[0].squeeze(0), pooled_output_01[1].squeeze(0)), dim=1)
                pooled_output_02 = torch.cat((pooled_output_02[0].squeeze(0), pooled_output_02[1].squeeze(0)), dim=1)

                pooled_output_01 = self.dense(pooled_output_01)
                pooled_output_02 = self.dense(pooled_output_02)

                #pooled_output_01 = encoded_layer_12_emb01.mean(dim=1)
                #pooled_output_02 = encoded_layer_12_emb02.mean(dim=1)

                #pooled_output_01 = self.dense(pooled_output_01)
                #pooled_output_02 = self.dense(pooled_output_02)

                pooled_output_01 = self.activation(pooled_output_01)
                pooled_output_02 = self.activation(pooled_output_02)

                pooled_output_01 = self.dropout(pooled_output_01)
                pooled_output_02 = self.dropout(pooled_output_02)

                z_i = normalize(self.instance_projector(pooled_output_01), dim=1)
                z_j = normalize(self.instance_projector(pooled_output_02), dim=1)

                c_i = self.cluster_projector(pooled_output_01)
                c_j = self.cluster_projector(pooled_output_02)

                c_i = self.softmax(c_i)
                c_j = self.softmax(c_j)

                return z_i, z_j, c_i, c_j

            else:
                exit()

    def forward_cluster(self, batch, pretrain = True):
        if pretrain:
            input_ids, input_mask, segment_ids, label_ids = batch
            encoded_layer_12, pooled_output = self.bert(input_ids, segment_ids, input_mask,
                                                        output_all_encoded_layers=False)
            _, pooled_output = self.rnn(encoded_layer_12)
            pooled_output = torch.cat((pooled_output[0].squeeze(0), pooled_output[1].squeeze(0)), dim=1)
            pooled_output = self.dense(pooled_output)
            pooled_output = self.activation(pooled_output)
            pooled_output = self.dropout(pooled_output)

            #z = normalize(self.instance_projector(pooled_output), dim=1)
            c = self.classifier(pooled_output)
            #c = self.classifier(feats)
            c = self.softmax(c)

            return c, pooled_output

        else:
            input_ids, input_mask, segment_ids, label_ids = batch
            encoded_layer_12, pooled_output = self.bert(input_ids, segment_ids, input_mask,
                                                        output_all_encoded_layers=False)
            #pooled_output = encoded_layer_12.mean(dim=1)
            #pooled_output = self.dense(pooled_output)
            _, pooled_output = self.rnn(encoded_layer_12)
            pooled_output = torch.cat((pooled_output[0].squeeze(0), pooled_output[1].squeeze(0)), dim=1)
            pooled_output = self.dense(pooled_output)
            pooled_output = self.activation(pooled_output)
            pooled_output = self.dropout(pooled_output)

            #z = normalize(self.instance_projector(pooled_output), dim=1)
            c = self.classifier(pooled_output)
            c = self.softmax(c)

            return c, pooled_output




'''
        if feature_ext:
            return pooled_output
        elif mode == 'train':
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss
        else:
            return pooled_output, logits
'''
'''
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        c = self.softmax(logits)
'''
        #c = torch.argmax(c, dim=1)


class Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
