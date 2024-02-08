import argparse
import sys
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
import OpenMatch as om
from OpenMatch.data.datasets.my_bert_dataset_disen_classification import BertDataset
from OpenMatch.models.bert_disentangled import BertDisentangled
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def test(args, model, test_loader, device):
    pred_attributes = []
    true_attributes = []
    for test_batch in test_loader:
        query_id, attribute_label = test_batch['query_id'], test_batch['attribute_label']
        with torch.no_grad():
            if args.model == 'bert':
                batch_score, pred_attribute, pred_attribute_adv = model(test_batch['input_ids'].to(device), test_batch['input_mask'].to(device), test_batch['segment_ids'].to(device))
                # print(len(batch_score))
            elif args.model == 'roberta':
                batch_score, _ = model(test_batch['input_ids'].to(device), test_batch['input_mask'].to(device))

            else:
                batch_score, _ = model(test_batch['query_idx'].to(device), test_batch['query_mask'].to(device),
                                       test_batch['doc_idx'].to(device), test_batch['doc_mask'].to(device))


            
            pred_attribute = pred_attribute.detach().cpu().tolist()
            pred_attributes.extend(pred_attribute)

            attribute_label = attribute_label.detach().cpu().tolist()
            true_attributes.extend(attribute_label)

    pred_attributes_2 = [0 if  pred_attributes[i] <0.5 else 1 for i in range(len(pred_attributes))]
    # Attribute classifier evaluation
    print("Attribute classifier evaluation")
    accuracy = accuracy_score(true_attributes, pred_attributes_2)
    myclassification_report = classification_report(true_attributes, pred_attributes_2)
    conf_matrix = confusion_matrix(true_attributes, pred_attributes_2)

    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{myclassification_report}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    # Adversary Attribute classifier evaluation
                
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default='ranking')
    parser.add_argument('-model', type=str, default='bert')
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-test', action=om.utils.DictOrStr, default='./data/test_toy.jsonl')
    parser.add_argument('-vocab', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-ent_vocab', type=str, default='')
    parser.add_argument('-pretrain', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-checkpoint', type=str, default='./checkpoints/bert.bin')
    parser.add_argument('-res', type=str, default='./results/bert.trec')
    parser.add_argument('-mode', type=str, default='cls')
    parser.add_argument('-n_kernels', type=int, default=21)
    parser.add_argument('-max_query_len', type=int, default=32)
    parser.add_argument('-max_doc_len', type=int, default=256)
    parser.add_argument('-maxp', action='store_true', default=False)
    parser.add_argument('-batch_size', type=int, default=32)
    # sys.argv = ['inference_disentanglement.py', '-task', 'ranking', '-model', 'bert', '-max_input', '6', '-vocab',
    #             'sentence-transformers/msmarco-MiniLM-L6-cos-v5', '-pretrain', 'sentence-transformers/msmarco-MiniLM-L6-cos-v5',
    #             '-checkpoint', '/home/ir-bias/Shirin/journal_paper_1/checkpoints/penalty_disentanglement/minilm_penalty_disen_3M.bin', '-res',
    #               '/home/ir-bias/Shirin/journal_paper_1/reranked/penalty_disentanglement/215_queries/minilm_penalty_disen_3M.trec', '-max_query_len', '32', '-max_doc_len', '221', '-batch_size', '256',
    #                 '-test', 'queries=/home/ir-bias/Shirin/gender_disentanglement/data/target_queries/social_neutrals/msmarco/neutral_queries.tsv,docs=/home/ir-bias/Shirin/msmarco/data/collection.tsv,trec=/home/ir-bias/Shirin/gender_disentanglement/runs/run.215_societal_rekabsaz.dev.labelled.trec']
    args = parser.parse_args()
    args.model = args.model.lower()
    if args.model == 'bert':
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        print('reading test data...')
        if args.maxp:
            test_set = om.data.datasets.BertMaxPDataset(
                dataset=args.test,
                tokenizer=tokenizer,
                mode='test',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        else:
            test_set = BertDataset(
                dataset=args.test,
                tokenizer=tokenizer,
                mode='test',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
    elif args.model == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        print('reading test data...')
        test_set = om.data.datasets.RobertaDataset(
            dataset=args.test,
            tokenizer=tokenizer,
            mode='test',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )

    else:
        tokenizer = om.data.tokenizers.WordTokenizer(
            pretrained=args.vocab
        )
        print('reading test data...')
        test_set = om.data.datasets.Dataset(
            dataset=args.test,
            tokenizer=tokenizer,
            mode='test',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )

    test_loader = om.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8
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
            model = BertDisentangled(
                pretrained=args.pretrain,
                mode=args.mode,
                task=args.task
            )

    else:
        raise ValueError('model name error.')

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

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    test(args, model, test_loader, device)

if __name__ == "__main__":
    main()
