from util import *
import torch.utils.data as util_data
from torch.utils.data import Dataset



def set_seed(seed):
    random.seed(seed)
    np.random.seed(10)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
class Data:
    
    def __init__(self, args):
        # 随机初始化
        set_seed(args.seed)

        # 载入数据集的关键信息
        max_seq_lengths = {'clinc':30, 'banking':55, 'snips': 35, "HWU64":25}
        args.max_seq_length = max_seq_lengths[args.dataset]

        # 随机选取已知类和未知类(得到IND和OOD的类别list)
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.all_label_list = self.get_labels(self.data_dir)
        print("the numbers of all labels:", len(self.all_label_list))

        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))
        print("the numbers of IND labels:", len(self.known_label_list), self.n_known_cls)

        self.n_unknown_cls = len(self.all_label_list) - len(self.known_label_list)
        self.unknown_label_list = list(set(self.all_label_list).difference(set(self.known_label_list)))
        print("the numbers of OOD labels:", len(self.unknown_label_list), self.n_unknown_cls)
        print(self.unknown_label_list)
        for k in range(len(self.unknown_label_list)):
            print(self.unknown_label_list[k])


        if args.IND_ratio!=1.0:
            self.n_known_cls = round(len(self.known_label_list) * args.IND_ratio)
            self.known_label_list = list(np.random.choice(np.array(self.known_label_list), self.n_known_cls, replace=False))
            print("revised: the numbers of IND labels:", len(self.known_label_list), self.n_known_cls)

        self.num_labels = int(len(self.unknown_label_list))*2

        # 载入数据集(tsv文件的表格，二维列表形式)
        train_sets = self.get_datasets(self.data_dir, 'train')
        eval_sets = self.get_datasets(self.data_dir, 'eval')
        test_sets = self.get_datasets(self.data_dir, 'test')

        # 划分OOD和IND (至此，数据结构都是list，元素为数据集中的一行(也是一个小的list)，并且还是字符形式的)
        self.train_labeled_examples, self.train_unlabeled_examples = self.divide_datasets_2(train_sets, args)
        print('train_num_labeled_samples', len(self.train_labeled_examples))
        print('train_num_unlabeled_samples', len(self.train_unlabeled_examples))

        self.eval_labeled_examples, self.eval_unlabeled_examples = self.divide_datasets(eval_sets)
        print('eval_num_labeled_samples', len(self.eval_labeled_examples))
        print('eval_num_unlabeled_samples', len(self.eval_unlabeled_examples))

        self.test_labeled_examples, self.test_unlabeled_examples = self.divide_datasets(test_sets)
        print('test_num_samples', len(self.test_labeled_examples))
        print('test_num_unlabeled_samples', len(self.test_unlabeled_examples))

        # (此时仍然还是字符形式的)
        self.train_labeled_examples = self.get_samples(self.train_labeled_examples, args, "train")
        self.train_unlabeled_examples = self.get_samples(self.train_unlabeled_examples, args, "train")

        self.eval_labeled_examples = self.get_samples(self.eval_labeled_examples, args, "eval")
        self.eval_unlabeled_examples = self.get_samples(self.eval_unlabeled_examples, args, "eval")

        self.test_labeled_examples = self.get_samples(self.test_labeled_examples, args, "test")
        self.test_unlabeled_examples = self.get_samples(self.test_unlabeled_examples, args, "test")

        # 封装成dataloader格式(此时需要vectorization)
        self.train_labeled_dataloader = self.get_loader(self.train_labeled_examples,self.known_label_list,args,"train")
        self.train_unlabeled_dataloader = self.augment_loader(self.train_unlabeled_examples, self.unknown_label_list, args,"train")

        self.eval_labeled_dataloader = self.get_loader(self.eval_labeled_examples,self.known_label_list, args, "eval")
        self.eval_unlabeled_dataloader = self.get_loader(self.eval_unlabeled_examples,self.unknown_label_list, args, "eval")

        self.test_labeled_dataloader = self.get_loader(self.test_labeled_examples,self.known_label_list, args, "test")
        self.test_unlabeled_dataloader = self.get_loader(self.test_unlabeled_examples,self.unknown_label_list, args, "test")

    def get_labels(self, data_dir):
        """See base class."""
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(test['label']))
        return labels

    def get_datasets(self, data_dir, mode = 'train', quotechar=None):
        with open(os.path.join(data_dir, mode+".tsv"), "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            i=0
            for line in reader:
                if (i==0):
                    i+=1
                    continue
                line[0] = line[0].strip()
                lines.append(line)
            return lines


    def divide_datasets(self, origin_data):
        labeled_examples, unlabeled_examples = [], []
        for example in origin_data:
            if example[-1] in self.known_label_list:
                labeled_examples.append(example)
            elif example[-1] in self.unknown_label_list:
                unlabeled_examples.append(example)
        return labeled_examples, unlabeled_examples

    def divide_datasets_2(self, origin_data, args):
        train_labels = np.array([example[-1] for example in origin_data])
        train_labeled_ids = []
        for label in self.known_label_list:
            num = round(len(train_labels[train_labels == label]) * args.labeled_ratio)
            pos = list(np.where(train_labels == label)[0])
            train_labeled_ids.extend(random.sample(pos, num))

        labeled_examples, unlabeled_examples = [], []
        for idx, example in enumerate(origin_data):
            if idx in train_labeled_ids:
                labeled_examples.append(example)
            elif example[-1] in self.unknown_label_list:
                unlabeled_examples.append(example)

        return labeled_examples, unlabeled_examples



    def get_samples(self, labelled_examples, args, mode):
        content_list, labels_list = [], []
        for example in labelled_examples:
            text = example[0]
            label = example[-1]
            content_list.append(text)
            labels_list.append(label)

        data = OriginSamples(content_list,labels_list)

        return data


    def get_embedding(self, labelled_examples, label_list, args, mode="train"):
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        features = convert_examples_to_features(labelled_examples, label_list, args.max_seq_length,
                                                           tokenizer)
        data=[]
        for f in features:
            results={
                "input_ids":f.input_ids,
                "input_mask":f.input_mask,
                "segment_ids":f.segment_ids,
                "label_id":f.label_id
            }
            #print("input_ids",f.input_ids)
            #print("input_mask",f.input_mask)
            #print("segment_ids",f.segment_ids)
            #print("label_id",f.label_id)

            data.append(results)

        #input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        #input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        #segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        #label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        #print(input_ids)
        #exit()

        #data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        return data



    def get_loader(self, labelled_examples, label_list, args, mode="train"):
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        features = convert_examples_to_features(labelled_examples, label_list, args.max_seq_length, tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        if mode == 'train':
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.pre_train_batch_size)
        else:
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)

        return dataloader

    def augment_loader(self, unlabelled_examples, label_list, args, mode="train"):
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        features = convert_examples_to_features(unlabelled_examples, label_list, args.max_seq_length, tokenizer)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        if mode == 'train':
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)
        else:
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)

        return dataloader



class OriginSamples(Dataset):
    def __init__(self, train_x, train_y):
        assert len(train_y) == len(train_x)
        self.train_x = train_x
        self.train_y = train_y


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i
    '''
    if len(label_list)==15:
        label_map = {'smart_home': 0, 'spending_history': 1, 'tire_pressure': 2, 'lost_luggage': 3, 'cancel': 4, 'reset_settings': 5, 'where_are_you_from': 6, 'book_flight': 7, 'bill_due': 8,
                     'accept_reservations': 9, 'expiration_date': 10, 'timezone': 11, 'new_card': 12, 'cancel_reservation': 13, 'income': 14}
        print(label_map)
    '''
    '''
    if len(label_list) == 23:
        label_map = { "top_up_by_bank_transfer_charge": 0,
"card_acceptance": 1,
"Refund_not_showing_up":2,
"card_not_working":3,
"transfer_fee_charged":4,
"verify_top_up":5,
"unable_to_verify_identity":6,
"beneficiary_not_allowed":7,
"card_linking":8,
"supported_cards_and_currencies":9,
"getting_spare_card":10,
"transfer_into_account":11,
"receiving_money":12,
"card_payment_fee_charged":13,
"automatic_top_up":14,
"declined_transfer":15,
"direct_debit_payment_not_recognised":16,
"pending_transfer":17,
"failed_transfer":18,
"card_delivery_estimate":19,
"cancel_transfer":20,
"topping_up_by_card":21,
"exchange_rate":22,
}

        print(label_map)

    elif len(label_list) == 15:
        label_map = {"getting_spare_card": 0,
                     "card_not_working": 1,
                     "exchange_rate": 2,
                     "card_acceptance": 3,
                     "topping_up_by_card": 4,
                     "declined_transfer": 5,
                     "supported_cards_and_currencies": 6,
                     "direct_debit_payment_not_recognised": 7,
                     "failed_transfer": 8,
                     "cancel_transfer": 9,
                     "card_payment_fee_charged": 10,
                     "Refund_not_showing_up": 11,
                     "verify_top_up": 12,
                     "beneficiary_not_allowed": 13,
                     "transfer_fee_charged": 14}

        print(label_map)
    '''
    features = []
    content_list = examples.train_x
    label_list = examples.train_y

    for i in range(len(content_list)):
        tokens_a = tokenizer.tokenize(content_list[i])

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[label_list[i]]

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features
