import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model import *
from init_parameter import *
from dataloader import *
from pretrain import *
from util import *
from contrastive_loss import *
from baseline_0.kmeans import *
from baseline_0.CC import *
from pretrain_kt import *
from KT import *


class ModelManager:
    
    def __init__(self, args, data, pretrained_model=None):
        set_seed(args.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if pretrained_model is None:
            pretrained_model = BertForModel.from_pretrained(args.bert_model, num_labels = data.n_known_cls)
            pretrained_model.to(self.device)
            root_path = "pretrain_models"
            pretrain_dir = os.path.join(root_path, args.pretrain_dir)
            if os.path.exists(pretrain_dir):
                pretrained_model = self.restore_model(args, pretrained_model)
        self.pretrained_model = pretrained_model

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id           
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model.to(self.device)

        #self.num_labels = 23
        self.num_labels = data.n_unknown_cls

        print("novel_num_label",self.num_labels)
        self.model = BertForModel.from_pretrained(args.bert_model, num_labels = self.num_labels)

        #BertForModel.from_pretrained(args.bert_model, num_labels = data.n_unknown_cls)

        if args.pretrain:
            self.load_pretrained_model(args)

        if args.freeze_bert_parameters:
            self.freeze_parameters(self.model)

        #total = sum([param.nelement() for param in self.model.parameters() if param.requires_grad])
        #print("Number of parameter: % .2fM" % (total / 1e6))
        #exit()

        self.model.to(self.device)
        print(self.model)

        num_train_examples = len(data.train_unlabeled_examples.train_x)
        self.num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs
        
        self.optimizer = self.get_optimizer(args)

        self.criterion_Maskinstance = MaskInstanceLoss(args.train_batch_size, args.instance_temperature, self.device).to(
            self.device)
        self.criterion_instance = InstanceLoss(args.train_batch_size, args.instance_temperature, self.device).to(
            self.device)
        self.KNN_Instance = KNN_InstanceLoss(args.train_batch_size, args.instance_temperature, self.device, 1).to(
            self.device)
        self.criterion_cluster = ClusterLoss(self.num_labels, args.cluster_temperature, self.device).to(
            self.device)

        self.best_eval_score = 0
        self.centroids = None
        self.training_SC_epochs = {}

        self.test_results = None
        self.predictions = None
        self.true_labels = None

    def get_features_labels(self, dataloader, model, args):

        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                feature = model(batch, mode = 'feature-extract')

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels

    def predict_k(self, args, data):

        feats, _ = self.get_features_labels(data.train_unlabeled_dataloader, self.pretrained_model, args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = data.num_labels).fit(feats)
        y_pred = km.labels_

        pred_label_list = np.unique(y_pred)
        drop_out = len(feats) / data.num_labels
        print('drop',drop_out)

        cnt = 0
        for label in pred_label_list:
            num = len(y_pred[y_pred == label])
            if num < drop_out:
                cnt += 1

        num_labels = len(pred_label_list) - cnt
        print('pred_num',num_labels)

        return num_labels


    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                         lr = args.lr,
                         warmup = args.warmup_proportion,
                         t_total = self.num_train_optimization_steps)
        return optimizer

    def pca_visualization(self,x,y,predicted):
        label_list=[0,1,2,3,4,5,6,7,8,9]
        path = args.save_results_path
        pca_visualization(x, y, label_list, os.path.join(path, "pca_test.png"))
        pca_visualization(x, predicted, label_list, os.path.join(path, "pca_test_2.png"))

    def tsne_visualization(self,x,y,predicted):
        label_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        path = args.save_results_path
        TSNE_visualization(x, y, label_list, os.path.join(path, "pca_test_b2.png"))
        TSNE_visualization(x, predicted, label_list, os.path.join(path, "pca_test_2_b2.png"))

    def tsne_visualization_2(self,x,y,predicted, epoch=100):
        label_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        path = args.save_results_path
        TSNE_visualization(x, y, label_list, os.path.join(path, "DKT.png"))


    def evaluation_2(self, args, data):
        self.model.eval()
        eval_dataloader = data.test_unlabeled_dataloader
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)

        step = 0
        for batch in tqdm(eval_dataloader, desc="evaluation"):
            step += 1
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits, feat = self.model.forward_cluster(batch)
                total_features = torch.cat((total_features, feat))
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = total_logits.max(dim=1)
        x_feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        results = clustering_score(y_true, y_pred)
        print(results)

        ind, _ = hungray_aligment(y_true, y_pred)
        map_ = {i[0]: i[1] for i in ind}
        y_pred_aligned = np.array([map_[idx] for idx in y_pred])

        cm = confusion_matrix(y_true, y_pred_aligned)
        print(cm)

    def evaluation(self, args, data):

        self.model.eval()
        eval_dataloader = data.test_unlabeled_dataloader
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)

        step=0
        for batch in tqdm(eval_dataloader, desc="evaluation"):
            step+=1
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits, feat = self.model.forward_cluster(batch)
                total_features = torch.cat((total_features, feat))
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = total_logits.max(dim=1)
        x_feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        results = clustering_score(y_true, y_pred)
        score = metrics.silhouette_score(x_feats, y_pred)
        results["SC"] = score
        print(results)

        min_d, max_d, mean_d, intra_list = intra_distance(x_feats, y_true, self.num_labels)
        print(min_d, max_d, mean_d, intra_list)
        #self.analysis_results["intra_distance"] = mean_d
        min_d, max_d, mean_d, inter_list = inter_distance(x_feats, y_true, self.num_labels)
        print(min_d, max_d, mean_d, inter_list)

        a = []

        for i in range(self.num_labels):
            a.append(inter_list[i]/intra_list[i])


        '''
        
        hard_class_list = select_hard(y_true, y_pred)
        print(hard_class_list)
        hard_class_list.sort()
        print(hard_class_list)

        min_d, max_d, mean_d, intra_list = intra_distance(x_feats, y_true, self.num_labels)
        print(min_d, max_d, mean_d, intra_list)
        #self.analysis_results["intra_distance"] = mean_d
        min_d, max_d, mean_d, inter_list = inter_distance(x_feats, y_true, self.num_labels)
        print(min_d, max_d, mean_d, inter_list)

        hard_class_list = select_hard_2(intra_list, inter_list)
        print(hard_class_list)
        #hard_class_list.sort()
        #print(hard_class_list)
        a = [1.2720418840483885, 1.9555266750992137, 2.0659313732064493, 2.1558232027445836, 2.268445805201286]
        for i in range(len(a)):
            print(hard_class_list.index(a[i]))

        exit()
        '''
        #self.analysis_results["inter_distance"] = mean_d


        ind, _ = hungray_aligment(y_true, y_pred)
        map_ = {i[0]: i[1] for i in ind}
        y_pred_aligned = np.array([map_[idx] for idx in y_pred])

        cm = confusion_matrix(y_true, y_pred_aligned)
        #cm = cm/40
        print(cm)
        exit()

        #self.pca_visualization(x_feats, y_true, y_pred)

        #file = "./outputs/results.csv"
        #with open(file, "w") as f:
        #    f.write(results)
        #    f.write("ground_truth\t")
        #    f.write(y_true)
        #    f.write("prediction\t")
        #    f.write(y_pred)
        #f.close()
        #print(y_true)
        #print(len(y_pred[y_pred>0.5]))

        self.test_results = results

        return results

    def visualize_training(self, args, data):
        self.model.eval()
        eval_dataloader = data.train_unlabeled_dataloader
        total_features = torch.empty((0, 768)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)

        step = 0
        for batch in tqdm(eval_dataloader, desc="evaluation"):
            step += 1
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits, feat = self.model.forward_cluster(batch)
                total_features = torch.cat((total_features, feat))
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = total_logits.max(dim=1)
        total_features = normalize(total_features, dim=1)
        x_feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        # acc = round(accuracy_score(y_true, y_pred) * 100, 2)

        results = clustering_score(y_true, y_pred)
        print(results)
        self.train_results = results

        # self.pca_visualization(x_feats, y_true, y_pred)
        self.tsne_visualization_2(x_feats, y_true, y_pred)



    def eval(self, args, data, type):
        self.model.eval()
        eval_dataloader = data.eval_unlabeled_dataloader
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)

        step = 0
        for batch in tqdm(eval_dataloader, desc="evaluation"):
            step += 1
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits, feat = self.model.forward_cluster(batch)
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))
                total_features = torch.cat((total_features, feat))

        total_probs, total_preds = total_logits.max(dim=1)
        x_feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        score = metrics.silhouette_score(x_feats,y_pred)
        # acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        results = clustering_score(y_true, y_pred)
        #print(results)
        #self.test_results = results
        if type == 1:
            return results["ARI"]
        else:
            return score

    def training_process_eval(self, args, data, epoch):
        self.model.eval()
        eval_dataloader = data.train_unlabeled_dataloader
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)

        step = 0
        for batch in tqdm(eval_dataloader, desc="evaluation"):
            step += 1
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                logits, feat = self.model.forward_cluster(batch)
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))
                total_features = torch.cat((total_features, feat))

        total_probs, total_preds = total_logits.max(dim=1)
        total_features = normalize(total_features, dim=1)
        x_feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        results = clustering_score(y_true, y_pred)
        score = metrics.silhouette_score(x_feats, y_pred)
        #score = results["NMI"]
        # acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        self.training_SC_epochs["epoch:" + str(epoch)] = score

        #self.tsne_visualization_2(x_feats, y_true, y_pred, epoch)

        return score


    def train(self, args, data):

        best_score = 0
        best_model = None
        wait = 0
        e_step = 0

        #SC_score = self.training_process_eval(args, data, e_step)
        #e_step += 1
        #print(SC_score)

        train_dataloader_1 = data.train_unlabeled_dataloader

        #contrastive clustering
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            loss = 0
            self.model.train()
            step = 0
            loss_epoch = 0
            for step, batch in enumerate(tqdm(data.train_unlabeled_dataloader, desc="Pseudo-Training")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids_1, input_mask_1, segment_ids_1, label_ids_1 = batch

                z_i, z_j, c_i, c_j = self.model(batch, mode='contrastive-clustering')

                loss_instance = self.criterion_instance(z_i, z_j)
                #loss_instance = self.criterion_Maskinstance(z_i, z_j, c_i, c_j)
                #loss_instance = self.KNN_Instance(z_i, z_j, c_i, c_j)
                loss_cluster = self.criterion_cluster(c_i, c_j)
                loss = loss_instance + loss_cluster
                '''
                loss_step = loss_instance + loss_cluster

                if step%2 == 0:
                    loss = 0
                    loss = loss + loss_step
                    continue
                else:
                    loss = loss + loss_step
                '''

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                step+=1
                print(f"Step [{step}/{len(train_dataloader_1)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
                loss_epoch += loss.item()
            print(f"Epoch [{epoch}/{args.num_train_epochs}]\t Loss: {loss_epoch / len(train_dataloader_1)}")

            #SC_score = self.training_process_eval(args, data, e_step)
            #e_step += 1
            #print(SC_score)

            eval_acc = self.eval(args, data, 0)
            print(eval_acc)
            if eval_acc > best_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                best_score = eval_acc
            else:
                wait += 1
                if wait >= args.wait_patient:
                    self.model = best_model
                    break

        self.model = best_model
        if args.save_model:
            self.save_model(args)

    def save_model(self, args):
        root_path = "DKT_models"
        pretrain_dir = os.path.join(root_path, "DKT_0.8")
        if not os.path.exists(pretrain_dir):
            os.makedirs(pretrain_dir)
        self.save_model = self.model.module if hasattr(self.model, 'module') else self.model
        model_file = os.path.join(pretrain_dir, WEIGHTS_NAME)
        model_config_file = os.path.join(pretrain_dir, CONFIG_NAME)
        torch.save(self.save_model.state_dict(), model_file)
        with open(model_config_file, "w") as f:
            f.write(self.save_model.config.to_json_string())

    def load_pretrained_model(self, args):
        pretrained_dict = self.pretrained_model.state_dict()
        classifier_params = ['classifier.weight', 'classifier.bias', 'cluster_projector.2.weight', 'cluster_projector.2.bias']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)
        

    def restore_model(self, args, model):
        root_path = "pretrain_models"
        pretrain_dir = os.path.join(root_path, args.pretrain_dir)
        output_model_file = os.path.join(pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model
    
    def freeze_parameters(self,model):
        for name, param in model.bert.named_parameters():  
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.method, args.known_cls_ratio, args.cluster_num_factor, args.seed, args.train_batch_size, args.lr, self.num_labels]
        names = ['dataset', 'method', 'known_cls_ratio', 'cluster_num_factor','seed', 'train_batch_size', 'learning_rate', 'K']
        vars_dict = {k:v for k,v in zip(names, var) }
        results = dict(self.test_results,**vars_dict)
        keys = list(results.keys())
        values = list(results.values())
        
        file_name = 'reselts_check_v3_K1.csv'
        results_path = os.path.join(args.save_results_path, file_name)
        
        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori,columns = keys)
            df1.to_csv(results_path,index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results,index=[1])
            df1 = df1.append(new,ignore_index=True)
            df1.to_csv(results_path,index=False)
        data_diagram = pd.read_csv(results_path)
        
        print('test_results', data_diagram)
        #self.save_training_process(args)

    def save_training_process(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        results = dict(self.training_SC_epochs)
        keys = list(results.keys())
        values = list(results.values())

        file_name = 'results_analysis_V2_100_trainigEpoch.csv'
        results_path = os.path.join(args.save_results_path, file_name)

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results, index=[1])
            df1 = df1.append(new, ignore_index=True)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)

        print('training_process_dynamic:', data_diagram)

if __name__ == '__main__':

    print('Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    data = Data(args)


    if args.pretrain:
        if args.method == "DKT":
            print('Pre-training begin...')
            #manager_p = PretrainModelManager(args, data)
            #manager_p.train(args, data)
            print('Pre-training finished!')
            #manager_p.load_models(args)
            #manager_p.analysis(args, data)
            #manager_p.save_results(args)
            #exit()
            #manager_p.evaluation(args, data)
            #exit()

            manager = ModelManager(args, data)
            print('Training begin...')
            manager.train(args, data)
            print('Training finished!')

            print('Evaluation begin...')
            manager.evaluation(args, data)
            print('Evaluation finished!')
            #manager.visualize_training(args, data)

            manager.save_results(args)

        if args.method == "kmeans":
            manager_Kmeans = KmeansModelManager(args, data)
            #manager_Kmeans.load_models(args)
            manager_Kmeans.BertForKmeans(args, data)
            manager_Kmeans.save_results_2(args)


        if args.method == "KT":
            print('Pre-training begin...')
            #manager_p = PretrainModelManager_KT(args, data)
            #manager_p.train(args, data)
            print('Pre-training finished!')
            #exit()

            manager = ModelManager_KT(args, data)
            print('Training begin...')
            manager.train(args, data)
            print('Training finished!')

            print('Evaluation begin...')
            manager.evaluation(args, data)
            print('Evaluation finished!')
            # manager.visualize_training(args, data)

            manager.save_results(args)

    else:
        if args.method == "kmeans":
            manager_Kmeans = KmeansModelManager(args, data)
            manager_Kmeans.BertForKmeans(args, data)
            manager_Kmeans.save_results(args)

        if args.method == "contrastive-clustering":
            manager_CC = CCModelManager(args, data)
            print('Training begin...')
            manager_CC.train(args, data)
            print('Training finished!')
            print('Evaluation begin...')
            manager_CC.evaluation(args, data)
            print('Evaluation finished!')
            manager_CC.visualize_training(args, data)
            manager_CC.save_results(args)

        #manager = ModelManager(args, data)






