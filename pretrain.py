from util import *
from model import *
from dataloader import *

class PretrainModelManager:
    
    def __init__(self, args, data):
        set_seed(args.seed)

        self.model = BertForModel.from_pretrained(args.bert_model, num_labels = data.n_known_cls)
        print(self.model)
        if args.freeze_bert_parameters:
            self.freeze_parameters(self.model)

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id           
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model.to(self.device)
        #n_gpu = torch.cuda.device_count()
        #if n_gpu > 1:
        #    self.model = torch.nn.DataParallel(self.model)

        self.num_train_optimization_steps = int(len(data.train_labeled_examples.train_x) / args.pre_train_batch_size) * args.num_pretrain_epochs
        
        self.optimizer = self.get_optimizer(args)
        
        self.best_eval_score = 0
        self.analysis_results = {}


    def load_models(self, args):
        print("loading models ....")
        self.model = self.restore_model_v2(args, self.model)

    def create_negative_dataset(self, data, args):
        negative_dataset = {}
        train_dataset = data.train_labeled_examples
        all_IND_data = data.get_embedding(train_dataset, data.known_label_list, args, "train")
        #print(all_IND_data)

        for line in all_IND_data:
            label = int(line["label_id"])
            inputs = line

            inputs.pop("label_id")
            if label not in negative_dataset.keys():
                negative_dataset[label] = [inputs]
            else:
                negative_dataset[label].append(inputs)

        #exit()
        return negative_dataset

    def generate_positive_sample(self, label: torch.Tensor):
        positive_num = self.positive_num

        # positive_num = 16
        positive_sample = []
        for index in range(label.shape[0]):
            input_label = int(label[index])
            positive_sample.extend(random.sample(self.negative_data[input_label], positive_num))

        return self.reshape_dict(positive_num, self.list_item_to_tensor(positive_sample))

    @staticmethod
    def list_item_to_tensor(inputs_list: List[Dict]):
        batch_list = {}
        for key, value in inputs_list[0].items():
            batch_list[key] = []
        for inputs in inputs_list:
            for key, value in inputs.items():
                batch_list[key].append(value)

        batch_tensor = {}
        for key, value in batch_list.items():
            batch_tensor[key] = torch.tensor(value)
        return batch_tensor

    def reshape_dict(self, sample_num, batch):
        """
        为什么要使用这个函数
        GPU在多卡训练是会对数据进行平均分配，
        当代码运行到最后一个batch时最后一张卡分到的数据比较少，但是negative_sample会平均分配，导致negative_sample和query_sample不匹配
        """
        for k, v in batch.items():
            shape = v.shape
            batch[k] = v.view([-1, sample_num, shape[-1]])
        return batch

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        return inputs

    def evaluation(self, args, data):
        self.model.eval()
        test_dataloader = data.test_labeled_dataloader
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.n_known_cls)).to(self.device)

        for batch in tqdm(test_dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                feat, logits = self.model(batch, mode='eval')
                total_features = torch.cat((total_features, feat))
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        print("accuracy:",acc)

        return acc


    def analysis(self, args, data):
        self.model.eval()
        test_dataloader = data.train_labeled_dataloader
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.n_known_cls)).to(self.device)

        for batch in tqdm(test_dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                feat, logits = self.model(batch, mode='eval')
                total_features = torch.cat((total_features, feat))
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        x_feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        min_d, max_d, mean_d, _ = intra_distance(x_feats, y_true, data.n_known_cls)
        self.analysis_results["intra_distance"] = mean_d
        min_d, max_d, mean_d, _ = inter_distance(x_feats, y_true, data.n_known_cls)
        self.analysis_results["inter_distance"] = mean_d
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        print("accuracy:", acc)

        return acc



    def eval(self, args, data):
        self.model.eval()
        total_features = torch.empty((0, 768)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.n_known_cls)).to(self.device)
        
        for batch in tqdm(data.eval_labeled_dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                feat, logits = self.model(batch, mode = 'eval')
                total_features = torch.cat((total_features, feat))
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim = 1)
        x_feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        acc = round(accuracy_score(y_true, y_pred) * 100, 2)

        return acc


    def train(self, args, data):  
 
        wait = 0
        best_model = None
        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(data.train_labeled_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                with torch.set_grad_enabled(True):
                    loss = self.model(batch, mode = "pre-trained")
                    loss.backward()
                    tr_loss += loss.item()
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
            
            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
            
            eval_score = self.eval(args, data)
            print('eval_score',eval_score)
            
            if eval_score > self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break
                
        self.model = best_model
        if args.save_model:
            self.save_model(args)

    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                         lr = args.lr_pre,
                         warmup = args.warmup_proportion,
                         t_total = self.num_train_optimization_steps)   
        return optimizer
    
    def save_model(self, args):
        root_path = "pretrain_models"
        pretrain_dir = os.path.join(root_path, args.pretrain_dir)
        if not os.path.exists(pretrain_dir):
            os.makedirs(pretrain_dir)
        self.save_model = self.model.module if hasattr(self.model, 'module') else self.model  
        model_file = os.path.join(pretrain_dir, WEIGHTS_NAME)
        model_config_file = os.path.join(pretrain_dir, CONFIG_NAME)
        torch.save(self.save_model.state_dict(), model_file)
        with open(model_config_file, "w") as f:
            f.write(self.save_model.config.to_json_string())

    def freeze_parameters(self,model):
        for name, param in model.bert.named_parameters():  
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def restore_model_v2(self, args, model):
        root_path = "pretrain_models"
        pretrain_dir = os.path.join(root_path, args.pretrain_dir)
        output_model_file = os.path.join(pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model

    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.seed,
               args.train_batch_size, args.lr, 8]
        names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor', 'seed',
                 'train_batch_size', 'learning_rate', 'K']
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(self.analysis_results, **vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        file_name = 'analysis_1.csv'
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

        print('test_results', data_diagram)