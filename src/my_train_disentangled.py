import argparse
import time
import random
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import OpenMatch as om
from OpenMatch.models.bert_disentangled import Bert
from OpenMatch.data.datasets.my_dataset_bias import Dataset
from OpenMatch.data.datasets.my_bert_dataset_disentangled import BertDataset
from OpenMatch.data.datasets.my_roberta_dataset_bias import RobertaDataset
from regularized_loss import bias_regularized_margin_ranking_loss
from entropy_loss import entropy_loss


def dev(args, model, metric, dev_loader, device):
    rst_dict = {}
    for dev_batch in dev_loader:
        query_id, doc_id, label, retrieval_score = dev_batch['query_id'], dev_batch['doc_id'], dev_batch['label'], dev_batch['retrieval_score']
        with torch.no_grad():
            if args.model == 'bert':
                batch_score, _, _ = model(dev_batch['input_ids'].to(device), dev_batch['input_mask'].to(device), dev_batch['segment_ids'].to(device))
            elif args.model == 'roberta':
                batch_score, _ = model(dev_batch['input_ids'].to(device), dev_batch['input_mask'].to(device))
            elif args.model == 'edrm':
                batch_score, _ = model(dev_batch['query_wrd_idx'].to(device), dev_batch['query_wrd_mask'].to(device),
                                       dev_batch['doc_wrd_idx'].to(device), dev_batch['doc_wrd_mask'].to(device),
                                       dev_batch['query_ent_idx'].to(device), dev_batch['query_ent_mask'].to(device),
                                       dev_batch['doc_ent_idx'].to(device), dev_batch['doc_ent_mask'].to(device),
                                       dev_batch['query_des_idx'].to(device), dev_batch['doc_des_idx'].to(device))
            else:
                batch_score, _ = model(dev_batch['query_idx'].to(device), dev_batch['query_mask'].to(device),
                                       dev_batch['doc_idx'].to(device), dev_batch['doc_mask'].to(device))
            if args.task == 'classification':
                batch_score = batch_score.softmax(dim=-1)[:, 1].squeeze(-1)
            batch_score = batch_score.detach().cpu().tolist()
            for (q_id, d_id, b_s, l) in zip(query_id, doc_id, batch_score, label):
                if q_id not in rst_dict:
                    rst_dict[q_id] = {}
                if d_id not in rst_dict[q_id] or b_s > rst_dict[q_id][d_id][0]:
                    rst_dict[q_id][d_id] = [b_s, l]
    return rst_dict


def train(args, model, loss_fn, m_optim, m_scheduler, adv_optim, adv_scheduler, metric, train_loader, dev_loader, device):
    
    best_mes = 0.0
    loss_attribute = nn.BCELoss().to(device)
    loss_adv = nn.BCELoss().to(device)
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.epoch):
        avg_loss = 0.0
        # data_iter = iter(train_loader)
        # batch = data_iter.__next__()
        for step, train_batch in enumerate(tqdm(train_loader)):
            # if step <20:
            #     print(train_batch["attribute_pos"])
            if args.model == 'bert':
                if args.task == 'ranking':
                    batch_score_pos, attribute_pos, adv_attribute_pos = model(train_batch['input_ids_pos'].to(device), train_batch['input_mask_pos'].to(device), train_batch['segment_ids_pos'].to(device))
                    batch_score_neg, attribute_neg, adv_attribute_neg = model(train_batch['input_ids_neg'].to(device), train_batch['input_mask_neg'].to(device), train_batch['segment_ids_neg'].to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device), train_batch['segment_ids'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'roberta':
                if args.task == 'ranking':
                    batch_score_pos, attribute_pos, adv_attribute_pos = model(train_batch['input_ids_pos'].to(device), train_batch['input_mask_pos'].to(device))
                    batch_score_neg, attribute_neg, adv_attribute_neg = model(train_batch['input_ids_neg'].to(device), train_batch['input_mask_neg'].to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'edrm':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                               train_batch['doc_pos_wrd_idx'].to(device), train_batch['doc_pos_wrd_mask'].to(device),
                                               train_batch['query_ent_idx'].to(device), train_batch['query_ent_mask'].to(device),
                                               train_batch['doc_pos_ent_idx'].to(device), train_batch['doc_pos_ent_mask'].to(device),
                                               train_batch['query_des_idx'].to(device), train_batch['doc_pos_des_idx'].to(device))
                    batch_score_neg, _ = model(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                               train_batch['doc_neg_wrd_idx'].to(device), train_batch['doc_neg_wrd_mask'].to(device),
                                               train_batch['query_ent_idx'].to(device), train_batch['query_ent_mask'].to(device),
                                               train_batch['doc_neg_ent_idx'].to(device), train_batch['doc_neg_ent_mask'].to(device),
                                               train_batch['query_des_idx'].to(device), train_batch['doc_neg_des_idx'].to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                           train_batch['doc_wrd_idx'].to(device), train_batch['doc_wrd_mask'].to(device),
                                           train_batch['query_ent_idx'].to(device), train_batch['query_ent_mask'].to(device),
                                           train_batch['doc_ent_idx'].to(device), train_batch['doc_ent_mask'].to(device),
                                           train_batch['query_des_idx'].to(device), train_batch['doc_des_idx'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            else:
                if args.task == 'ranking':
                    batch_score_pos, attribute_pos, adv_attribute_pos = model(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                               train_batch['doc_pos_idx'].to(device), train_batch['doc_pos_mask'].to(device))
                    batch_score_neg, attribute_neg, adv_attribute_neg = model(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                               train_batch['doc_neg_idx'].to(device), train_batch['doc_neg_mask'].to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                           train_batch['doc_idx'].to(device), train_batch['doc_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            if args.task == 'ranking':
                ranking_loss = loss_fn(batch_score_pos.tanh(), batch_score_neg.tanh(), torch.ones(batch_score_pos.size()).to(device))
                # ranking_loss = bias_regularized_margin_ranking_loss(batch_score_pos.tanh(), batch_score_neg.tanh(),
                #                                                   args.regularizer,
                #                                                   train_batch["bias_neg"].to(device))
                # loss_attribute                
                attribute_loss_pos = loss_attribute(torch.sigmoid(attribute_pos), train_batch["attribute_pos"].to(device))
                attribute_loss_neg = loss_attribute(torch.sigmoid(attribute_neg), train_batch["attribute_neg"].to(device))

                # loss_adv
                # batch_loss_adv_pos = loss_adv(torch.sigmoid(adv_attribute_pos), train_batch["attribute_pos"].to(device))
                # batch_loss_adv_neg = loss_adv(torch.sigmoid(adv_attribute_neg), train_batch["attribute_neg"].to(device))

                # entropy_loss
                # hloss_pos = entropy_loss(torch.sigmoid(adv_attribute_pos))
                # hloss_neg = entropy_loss(torch.sigmoid(adv_attribute_neg))
               #  print(entropy_loss(torch.sigmoid(adv_attribute_pos)))
               #  print(entropy_loss(torch.sigmoid(adv_attribute_neg)))
                # total losses
                batch_loss = ranking_loss + attribute_loss_pos + attribute_loss_neg # + 0.001 * hloss_pos + 0.001 * hloss_neg
                # batch_loss_adv = batch_loss_adv_pos + batch_loss_adv_neg

            elif args.task == 'classification':
                batch_loss = loss_fn(batch_score, train_batch['label'].to(device))
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
            if torch.cuda.device_count() > 1:
                batch_loss = batch_loss.mean()
            avg_loss += batch_loss.item()

            # batch_loss_adv.backward(retain_graph=True)
            # adv_optim.step()
            # adv_scheduler.step()
            # adv_optim.zero_grad()

            batch_loss.backward()
            m_optim.step()
            m_scheduler.step()
            m_optim.zero_grad()

            if (step+1) % args.eval_every == 0:
                with torch.no_grad():
                    rst_dict = dev(args, model, metric, dev_loader, device)
                    om.utils.save_trec(args.res, rst_dict)
                    if args.metric.split('_')[0] == 'mrr':
                        mes = metric.get_mrr(args.qrels, args.res, args.metric)
                    else:
                        mes = metric.get_metric(args.qrels, args.res, args.metric)
                if mes >= best_mes:
                    best_mes = mes
                    print('save_model...')
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(), args.save)
                    else:
                        torch.save(model.state_dict(), args.save)
                print(step+1, avg_loss/args.eval_every, mes, best_mes)
                avg_loss = 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default='ranking')
    parser.add_argument('-model', type=str, default='bert')
    parser.add_argument('-reinfoselect', action='store_true', default=False)
    parser.add_argument('-reset', action='store_true', default=False)
    parser.add_argument('-train', action=om.utils.DictOrStr, default='./data/train_rank_toy.jsonl')
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-save', type=str, default='./checkpoints/bert.bin')
    parser.add_argument('-dev', action=om.utils.DictOrStr, default='./data/dev_toy.jsonl')
    parser.add_argument('-qrels', type=str, default='./data/qrels_toy')
    parser.add_argument('-vocab', type=str, default='allenai/scibert_scivocab_uncased/vocab.txt')
    parser.add_argument('-ent_vocab', type=str, default='')
    parser.add_argument('-pretrain', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-checkpoint', type=str, default=None)
    parser.add_argument('-res', type=str, default='./results/bert.trec')
    parser.add_argument('-metric', type=str, default='ndcg_cut_10')
    parser.add_argument('-mode', type=str, default='cls')
    parser.add_argument('-n_kernels', type=int, default=21)
    parser.add_argument('-max_query_len', type=int, default=32)
    parser.add_argument('-max_doc_len', type=int, default=221)
    parser.add_argument('-maxp', action='store_true', default=False)
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-lr', type=float, default=3e-6)
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-n_warmup_steps', type=int, default=16000)
    parser.add_argument('-eval_every', type=int, default=10000)
    parser.add_argument('-regularizer', type=float, default=1)
    # sys.argv = ["my_train_disentangled.py", "-task", "ranking", "-model", "bert", "-train", "/home/ir-bias/Shirin/journal_paper_1/data/bias_dataset_penalty+disentanglement_6M.tsv",
    #             "-dev", "/home/ir-bias/Shirin/journal_paper_1/data/dev.100.jsonl", "-save", "/home/ir-bias/Shirin/journal_paper_1/checkpoints/penalty_disentanglement/", "-qrels", "/home/ir-bias/Shirin/journal_paper_1/runs/qrels.dev.tsv",
    #             "-vocab", "sentence-transformers/msmarco-MiniLM-L6-cos-v5", "-pretrain", "sentence-transformers/msmarco-MiniLM-L6-cos-v5", "-res", "/home/shirin/journal_paper_1/results/penalty_disentanglement/6M_minilm.trec",
    #             "-metric", "mrr_cut_10", "-batch_size", "16", "-max_input", "100"]
    args = parser.parse_args()

    # random.seed = 42
    # Set a random seed for PyTorch on both CPU and GPU (if available)
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    args.model = args.model.lower()
    if args.model == 'bert':
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        print('reading training data...')
        if args.maxp:
            train_set = om.data.datasets.BertMaxPDataset(
                dataset=args.train,
                tokenizer=tokenizer,
                mode='train',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        else:
            train_set = BertDataset(
                dataset=args.train,
                tokenizer=tokenizer,
                mode='train',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        print('reading dev data...')
        if args.maxp:
            dev_set = om.data.datasets.BertMaxPDataset(
                dataset=args.dev,
                tokenizer=tokenizer,
                mode='dev',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        else:
            dev_set = om.data.datasets.BertDataset(
                dataset=args.dev,
                tokenizer=tokenizer,
                mode='dev',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
               task=args.task
            )
    elif args.model == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        print('reading training data...')
        train_set = RobertaDataset(
            dataset=args.train,
            tokenizer=tokenizer,
            mode='train',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )
        print('reading dev data...')
        dev_set = om.data.datasets.RobertaDataset(
            dataset=args.dev,
            tokenizer=tokenizer,
            mode='dev',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )
    elif args.model == 'edrm':
        tokenizer = om.data.tokenizers.WordTokenizer(
            pretrained=args.vocab
        )
        ent_tokenizer = om.data.tokenizers.WordTokenizer(
            vocab=args.ent_vocab
        )
        print('reading training data...')
        train_set = om.data.datasets.EDRMDataset(
            dataset=args.train,
            wrd_tokenizer=tokenizer,
            ent_tokenizer=ent_tokenizer,
            mode='train',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            des_max_len=20,
            max_ent_num=3,
            max_input=args.max_input,
            task=args.task
        )
        print('reading dev data...')
        dev_set = om.data.datasets.EDRMDataset(
            dataset=args.dev,
            wrd_tokenizer=tokenizer,
            ent_tokenizer=ent_tokenizer,
            mode='dev',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            des_max_len=20,
            max_ent_num=3,
            max_input=args.max_input,
            task=args.task
        )
    else:
        tokenizer = om.data.tokenizers.WordTokenizer(
            pretrained=args.vocab
        )
        print('reading training data...')
        train_set = Dataset(
            dataset=args.train,
            tokenizer=tokenizer,
            mode='train',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )
        print('reading dev data...')
        dev_set = om.data.datasets.Dataset(
            dataset=args.dev,
            tokenizer=tokenizer,
            mode='dev',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )
    train_loader = om.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1
    )
    dev_loader = om.data.DataLoader(
        dataset=dev_set,
        batch_size=args.batch_size * 16,
        shuffle=False,
        num_workers=1
    )

    if args.model == 'bert' or args.model == 'roberta':
        if args.maxp:
            model = om.models.BertMaxP(
                pretrained=args.pretrain,
                max_query_len=args.max_query_len,
                max_doc_len=args.max_doc_len,
                mode=args.mode,
                task=args.task
            )
        else:
            model = Bert(
                pretrained=args.pretrain,
                mode=args.mode,
                task=args.task
            )
        if args.reinfoselect:
            policy = om.models.Bert(
                pretrained=args.pretrain,
                mode=args.mode,
                task='classification'
            )
    elif args.model == 'edrm':
        model = om.models.EDRM(
            wrd_vocab_size=tokenizer.get_vocab_size(),
            ent_vocab_size=ent_tokenizer.get_vocab_size(),
            wrd_embed_dim=tokenizer.get_embed_dim(),
            ent_embed_dim=128,
            max_des_len=20,
            max_ent_num=3,
            kernel_num=args.n_kernels,
            kernel_dim=128,
            kernel_sizes=[1, 2, 3],
            wrd_embed_matrix=tokenizer.get_embed_matrix(),
            ent_embed_matrix=None,
            task=args.task
        )
    elif args.model == 'tk':
        model = om.models.TK(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            head_num=10,
            hidden_dim=100,
            layer_num=2,
            kernel_num=args.n_kernels,
            dropout=0.0,
            embed_matrix=tokenizer.get_embed_matrix(),
            task=args.task
        )
    elif args.model == 'cknrm':
        model = om.models.ConvKNRM(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            kernel_num=args.n_kernels,
            kernel_dim=128,
            kernel_sizes=[1, 2, 3],
            embed_matrix=tokenizer.get_embed_matrix(),
            task=args.task
        )
    elif args.model == 'knrm':
        model = om.models.KNRM(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            kernel_num=args.n_kernels,
            embed_matrix=tokenizer.get_embed_matrix(),
            task=args.task
        )
    else:
        raise ValueError('model name error.')

    if args.reinfoselect and args.model != 'bert':
        policy = om.models.ConvKNRM(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            kernel_num=args.n_kernels,
            kernel_dim=128,
            kernel_sizes=[1, 2, 3],
            embed_matrix=tokenizer.get_embed_matrix(),
            task='classification'
        )

    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint)
        if args.model == 'bert':
            st = {}
            for k in state_dict:
                if k.startswith('bert'):
                    st['_model'+k[len('bert'):]] = state_dict[k]
                elif k.startswith('classifier'):
                    st['_dense'+k[len('classifier'):]] = state_dict[k]
                else:
                    st[k] = state_dict[k]
            model.load_state_dict(st)
        else:
            model.load_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.reinfoselect:
        if args.task == 'ranking':
            loss_fn = nn.MarginRankingLoss(margin=1, reduction='none')
        elif args.task == 'classification':
            loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            raise ValueError('Task must be `ranking` or `classification`.')
    else:
        if args.task == 'ranking':
            loss_fn = nn.MarginRankingLoss(margin=1)
        elif args.task == 'classification':
            loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError('Task must be `ranking` or `classification`.')
        
    # define the optimizers
    m_parameters = [p for p in model._model.parameters()]+[p for p in model._ranking.parameters()]+[p for p in model._attribute.parameters()]
    # m_optim = torch.optim.Adam(m_parameters, lr=args.lr)
    m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, m_parameters), lr=args.lr)
    m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=args.n_warmup_steps, num_training_steps=len(train_set)*args.epoch//args.batch_size)
    # adv_parameters = list(model._adv_attribute.parameters())
    adv_parameters = [p for p in model._adv_attribute.parameters()]
    adv_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, adv_parameters), lr=args.lr)
    adv_scheduler = get_linear_schedule_with_warmup(adv_optim, num_warmup_steps=args.n_warmup_steps, num_training_steps=len(train_set)*args.epoch//args.batch_size)
    if args.reinfoselect:
        p_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, policy.parameters()), lr=args.lr)
    metric = om.metrics.Metric()

    model.to(device)
    if args.reinfoselect:
        policy.to(device)
    loss_fn.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        loss_fn = nn.DataParallel(loss_fn)

    if args.reinfoselect:
        print('reinforce not supported for the bias-aware version')
    else:
        time1 = time.time()
        print("start training")
        print("regularizer: ", args.regularizer)
        train(args, model, loss_fn, m_optim, m_scheduler, adv_optim, adv_scheduler, metric, train_loader, dev_loader, device)
        time2 = time.time()
        print("training time = {}".format(time2-time1))


if __name__ == "__main__":
    main()
