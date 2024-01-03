import argparse
import os
import sys
from tqdm import tqdm
import pdb

from document_neutrality import DocumentNeutrality

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--collection-path', action='store', dest='collection_path',
                    default="../../data/bias_dataset.tsv",
                    help='path the the collection file in tsv format (docid [tab] doctext)')
parser.add_argument('--representative-words-path', action='store', dest='representative_words_path',
                    default="wordlist_gender_representative.txt",
                    help='path to the list of representative words which define the protected attribute')
parser.add_argument('--threshold', action='store', type=int, default=1,
                    help='threshold on the number of sensitive words')
parser.add_argument('--out-file', action='store', dest='out_file', 
                    default="../../data/dataset_fairness.tsv",
                    help='output file containing docids and document neutrality scores')

args = parser.parse_args()




doc_neutrality = DocumentNeutrality(representative_words_path=args.representative_words_path,
                                  threshold=args.threshold,
                                  groups_portion={'f':0.5, 'm':0.5})
max_samples = 12800000
# max_samples = 10
with open(args.out_file, "w", encoding="utf8") as fw:
    with open(args.collection_path, "r", encoding="utf8") as fr:
        for n, line in tqdm(enumerate(fr)):
            if n %1000 ==0:
                print(n)
            if n > max_samples:
                break
            vals = line.strip().split('\t')
            query = vals[0]
            doc_pos = vals[1]
            doc_neg = vals[2]
            
            doc_pos_tokens = doc_pos.lower().split(' ') # it is expected that the input document is already cleaned and pre-tokenized
            doc_neg_tokens = doc_neg.lower().split(' ') # it is expected that the input document is already cleaned and pre-tokenized

            _neutrality_pos = doc_neutrality.get_neutrality(doc_pos_tokens)
            _neutrality_neg = doc_neutrality.get_neutrality(doc_neg_tokens)

            fw.write("%s\t%s\t%s\t%f\t%f\n" % (query, doc_pos, doc_neg, _neutrality_pos, _neutrality_neg))



        



            