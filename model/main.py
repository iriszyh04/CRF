import os
import logging
import numpy as np
import torch
from utils import commonUtils, metricsUtils, decodeUtils, trainUtils
import config
import dataset
import albertcrf_model  # 使用AlbertNerModel
from torch.utils.data import DataLoader, RandomSampler
from transformers import AlbertTokenizer  # 使用 AlbertTokenizer
from tensorboardX import SummaryWriter
project_root = os.path.abspath('C:/Users/J/Desktop/try/CRF')
if torch.__version__.startswith("2."):
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    
args = config.Args().get_parser()
commonUtils.set_seed(args.seed)
logger = logging.getLogger(__name__)

special_model_list = ['bilstm', 'crf', 'idcnn']
tensorboard_log_dir = os.path.join(project_root, 'tensorboard')
if args.use_tensorboard == "True":
    writer = SummaryWriter(log_dir='./tensorboard')

class AlbertForNer:
    def __init__(self, args, train_loader, dev_loader, test_loader, idx2tag):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.args = args
        self.idx2tag = idx2tag
        # 使用 AlbertNerModel 代替 BertNerModel
        model = albertcrf_model.AlbertNerModel(args)
        self.model, self.device = trainUtils.load_model_and_parallel(model, args.gpu_ids)
        self.model.to(self.device)
        if torch.__version__.startswith("2."):
            self.model = torch.compile(self.model)
        self.t_total = len(self.train_loader) * args.train_epochs
        self.optimizer, self.scheduler = trainUtils.build_optimizer_and_scheduler(args, model, self.t_total)

    def train(self):
        # 训练代码
        global_step = 0
        self.model.zero_grad()
        eval_steps = 90  # 每多少个step打印损失及进行验证
        best_f1 = 0.0
        for epoch in range(self.args.train_epochs):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for key in batch_data.keys():
                    if key != 'texts':
                        batch_data[key] = batch_data[key].to(self.device)

                loss, logits = self.model(batch_data['token_ids'], batch_data['attention_masks'], batch_data['token_type_ids'], batch_data['labels'])

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

                logger.info('【train】 epoch:{} {}/{} loss:{:.4f}'.format(epoch, global_step, self.t_total, loss.item()))
                if self.args.use_tensorboard == "True":
                    writer.add_scalar('train/loss', loss.item(), global_step)
                global_step += 1
                if global_step % eval_steps == 0:
                    dev_loss, precision, recall, f1_score = self.dev()
                    if self.args.use_tensorboard == "True":
                        writer.add_scalar('dev/loss', dev_loss, global_step)
                    logger.info('[eval] loss:{:.4f} precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(dev_loss, precision, recall, f1_score))
                    if f1_score > best_f1:
                        trainUtils.save_model(self.args, self.model, model_name + '_' + args.data_name, global_step)
                        best_f1 = f1_score

    def dev(self):
        # 验证集评估代码
        self.model.eval()
        with torch.no_grad():
            batch_output_all = []
            tot_dev_loss = 0.0
            for eval_step, dev_batch_data in enumerate(self.dev_loader):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(self.device)
                dev_loss, dev_logits = self.model(dev_batch_data['token_ids'], dev_batch_data['attention_masks'], dev_batch_data['token_type_ids'], dev_batch_data['labels'])
                tot_dev_loss += dev_loss.item()
                batch_output = dev_logits.detach().cpu().numpy()
                batch_output = np.argmax(batch_output, axis=2).tolist()

                if len(batch_output_all) == 0:
                    batch_output_all = batch_output
                else:
                    batch_output_all = batch_output_all + batch_output

            total_count = [0 for _ in range(len(label2id))]
            role_metric = np.zeros([len(id2label), 3])
            for pred_label, tmp_callback in zip(batch_output_all, dev_callback_info):
                text, gt_entities = tmp_callback
                tmp_metric = np.zeros([len(id2label), 3])
                pred_entities = decodeUtils.bioes_decode(pred_label[1:1 + len(text)], text, self.idx2tag)
                for idx, _type in enumerate(label_list):
                    if _type not in pred_entities:
                        pred_entities[_type] = []
                    total_count[idx] += len(gt_entities[_type])
                    tmp_metric[idx] += metricsUtils.calculate_metric(gt_entities[_type], pred_entities[_type])

                role_metric += tmp_metric

            mirco_metrics = np.sum(role_metric, axis=0)
            mirco_metrics = metricsUtils.get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
            return tot_dev_loss, mirco_metrics[0], mirco_metrics[1], mirco_metrics[2]

    def test(self, model_path, test_callback_info=None):
        model = albertcrf_model.AlbertNerModel(self.args)
        model, device = trainUtils.load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.to(device)
        model.eval()
        pred_label = []
        with torch.no_grad():
            for eval_step, dev_batch_data in enumerate(self.test_loader):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(device)
                _, logits = model(dev_batch_data['token_ids'], dev_batch_data['attention_masks'], dev_batch_data['token_type_ids'], dev_batch_data['labels'])
                batch_output = logits.detach().cpu().numpy()
                batch_output = np.argmax(batch_output, axis=2).tolist()

                if len(pred_label) == 0:
                    pred_label = batch_output
                else:
                    pred_label = pred_label + batch_output

            total_count = [0 for _ in range(len(id2label))]
            role_metric = np.zeros([len(id2label), 3])
            if test_callback_info is None:
                test_callback_info = dev_callback_info
            for pred, tmp_callback in zip(pred_label, test_callback_info):
                text, gt_entities = tmp_callback
                tmp_metric = np.zeros([len(id2label), 3])
                pred_entities = decodeUtils.bioes_decode(pred[1:1 + len(text)], text, self.idx2tag)
                for idx, _type in enumerate(label_list):
                    if _type not in pred_entities:
                        pred_entities[_type] = []
                    total_count[idx] += len(gt_entities[_type])
                    tmp_metric[idx] += metricsUtils.calculate_metric(gt_entities[_type], pred_entities[_type])

                role_metric += tmp_metric
            logger.info(metricsUtils.classification_report(role_metric, label_list, id2label, total_count))

    def predict(self, raw_text, model_path):
        model = albertcrf_model.AlbertNerModel(self.args)
        model, device = trainUtils.load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.to(device)
        model.eval()
        with torch.no_grad():
            tokenizer = AlbertTokenizer.from_pretrained(self.args.bert_dir)
            tokens = [i for i in raw_text]
            encode_dict = tokenizer.encode_plus(
                text=tokens,
                max_length=self.args.max_seq_len,
                padding='max_length',
                truncation='longest_first',
                is_pretokenized=True,
                return_token_type_ids=True,
                return_attention_mask=True
            )
            token_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).long().unsqueeze(0).to(device)
            attention_masks = torch.from_numpy(np.array(encode_dict['attention_mask'])).long().unsqueeze(0).to(device)
            token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'])).long().unsqueeze(0).to(device)

            logits = model(token_ids, attention_masks, token_type_ids, None)
            output = logits.detach().cpu().numpy()
            output = np.argmax(output, axis=2)
            pred_entities = decodeUtils.bioes_decode(output[0][1:1 + len(tokens)], "".join(tokens), self.idx2tag)
            logger.info(pred_entities)


if __name__ == '__main__':
    data_name = args.data_name
    model_name = args.model_name

    model_name_dict = {
        ("True", "False", "True"): '{}_bilstm_crf'.format(model_name),
        ("True", "False", "False"): '{}_bilstm'.format(model_name),
        ("False", "False", "False"): '{}'.format(model_name),
        ("False", "False", "True"): '{}_crf'.format(model_name),
        ("False", "True", "True"): '{}_idcnn_crf'.format(model_name),
        ("False", "True", "False"): '{}_idcnn'.format(model_name),
    }

    # 根据模型名称选择
    if args.model_name == 'bilstm':
        args.use_lstm = "True"
        args.use_idcnn = "False"
        args.use_crf = "True"
        model_name = "bilstm_crf"
    elif args.model_name == 'crf':
        model_name = "crf"
        args.use_lstm = "False"
        args.use_idcnn = "False"
        args.use_crf = "True"
    elif args.model_name == "idcnn":
        args.use_idcnn = "True"
        args.use_lstm = "False"
        args.use_crf = "True"
        model_name = "idcnn_crf"
    else:
        if args.use_lstm == "True" and args.use_idcnn == "True":
            raise Exception("请不要同时使用bilstm和idcnn")
        model_name = model_name_dict[(args.use_lstm, args.use_idcnn, args.use_crf)]

    args.data_name = data_name
    args.model_name = model_name
    commonUtils.set_logger(os.path.join(args.log_dir, '{}_{}.log'.format(model_name, args.data_name)))

    # 使用修改后的 AlbertForNer
    if data_name == "cner":
        args.data_dir = './data/cner'
        data_path = os.path.join(args.data_dir, 'final_data')

        ent2id_dict = commonUtils.read_json(other_path, 'nor_ent2id')
        label_list = commonUtils.read_json(other_path, 'labels')
        label2id = {}
        id2label = {}
        for k, v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        query2id = {}
        id2query = {}
        for k, v in ent2id_dict.items():
            query2id[k] = v
            id2query[v] = k
        logger.info(id2query)
        args.num_tags = len(ent2id_dict)
        logger.info(args)

        train_features, train_callback_info = commonUtils.read_pkl(data_path, 'train')
        train_dataset = dataset.NerDataset(train_features)
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  sampler=train_sampler,
                                  num_workers=2)
        dev_features, dev_callback_info = commonUtils.read_pkl(data_path, 'dev')
        dev_dataset = dataset.NerDataset(dev_features)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.eval_batch_size,
                                num_workers=2)
        test_features, test_callback_info = commonUtils.read_pkl(data_path, 'test')
        test_dataset = dataset.NerDataset(test_features)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=args.eval_batch_size,
                                 num_workers=2)

        # 将配置参数都保存下来
        commonUtils.save_json('./checkpoints/{}_{}/'.format(model_name, args.data_name), vars(args), 'args')
        AlbertForNer = AlbertForNer(args, train_loader, dev_loader, test_loader, id2query)
        AlbertForNer.train()

