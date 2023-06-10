import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from model import DenseEncoder, SparseEncoder, FullyConnectedNetwork
from bert import Bert
from data import ContrastiveDataset, SingleDataset
from optim import get_optimizer
import numpy as np
from utils import bool_flag, get_logger, bin_tensor, get_most_free_gpu
import setproctitle
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import pandas as pd
from losses import MASE
import copy
import json


class Trainer:

    def __init__(self, args):

        self.args = args
        torch.manual_seed(self.args.seed)

        if not torch.cuda.is_available(): args.device = "cpu"
        if self.args.process_name is None:
            process_name = f'{args.exp_name}'
        else:
            process_name = self.args.process_name
        setproctitle.setproctitle(process_name)

        exp_folder = Path(args.exp_folder)
        exp_folder.mkdir(exist_ok=True)
        dump_path = exp_folder / args.exp_name
        dump_path.mkdir(exist_ok=True)
        self.dump_path = dump_path
        args.dump_path = str(dump_path)
        with open(dump_path / 'args.pkl', 'wb') as f:
            torch.save(args, f)

        #if self.args.device == 'cpu':
        #    self.device = args.device
        #else:
        #    gpu_index = get_most_free_gpu()
        #    os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu_index
        #    self.device = f'cuda:{gpu_index}'
        self.device = self.args.device
        self.logger = get_logger(dump_path / 'train.log')
        self.logger.info(str(args))

        ### DATASET

        self.logger.info("Building dataset")

        train_df = pd.read_csv(args.data_path+'merfish_mouse_cortex.csv')
        train_df = train_df.sort_values('region')
        self.train_gene_names = list(train_df.columns[1:-8])
        self.selected_genes = []
        if args.selected_genes_path != None:
            with open(args.selected_genes_path,'r') as f:
                self.selected_genes = json.load(f) 
        self.n_classes = len(train_df['subclass'].unique())
        self.subclass2id = {train_df['subclass'].unique()[i]:i for i in range(len(train_df['subclass'].unique()))}

        if self.args.eval_ood:
            ood = pd.read_csv(self.args.data_path+'ood_mousehippocampus.csv')
            if self.args.loss == 'classification':
                ood = ood[ood['subclass'].isin(self.subclass2id.keys())]
            self.ood_gene_names = list(ood.columns[:-3])
        else:
            self.ood_gene_names = []
        #self.ood2 = pd.read_csv(self.args.data_path+'ood_mouseembryo.csv')

        loaded_embeddings = torch.load(args.data_path+'mouse_embedding.torch')
        self.embedding_gene_names = loaded_embeddings.keys()
        self.train_gene_names = list(set(self.train_gene_names).intersection(self.embedding_gene_names))
        self.ood_gene_names   = list(set(self.ood_gene_names).intersection(self.embedding_gene_names))  
        self.total_gene_names = list(set(self.train_gene_names).union(set(self.ood_gene_names)))
        # sort ood_gene_names so that train_gene_names appear first, then selected genes and then the others
        def key_fn(name):
            if name in self.train_gene_names:
                return self.train_gene_names.index(name)
            elif name in self.selected_genes and len(self.selected_genes) != 0:
                return len(self.train_gene_names) + self.selected_genes.index(name)
            else:
                return np.inf
        self.ood_gene_names = sorted(self.ood_gene_names, key=key_fn)
        self.total_gene_names = sorted(self.total_gene_names, key=key_fn)
        if self.args.model == 'mlp':
            self.total_gene_names = self.train_gene_names
        self.n_genes_train = len(self.train_gene_names)
        self.n_genes_total = len(self.total_gene_names)

        test_regions = train_df['region'].unique()[:3].tolist() # take smallest
        train_df, test_df = train_df.loc[~train_df['region'].isin(test_regions)], train_df.loc[train_df['region'].isin(test_regions)]
        test1 = test_df.loc[test_df['region']==test_df['region'].unique()[1]]
        test2 = test_df.loc[test_df['region']==test_df['region'].unique()[2]]

        self.id2gene = {i:self.total_gene_names[i] for i in range(self.n_genes_total)}
        self.id2gene[self.n_genes_total] = "PAD"
        self.gene2id = {v:k for k,v in self.id2gene.items()} 
        self.pad_idx = self.gene2id["PAD"]

        if self.args.loss != 'classification':
            self.dataset = ContrastiveDataset(train_df, self.train_gene_names, self.gene2id, self.args)
        else:
            self.dataset = SingleDataset(train_df  , self.train_gene_names, self.gene2id, self.subclass2id, self.args, path="/scratch/baffou/pkl/10_bins/train_percentile.pkl")
        self.test1 =       SingleDataset(     test1, self.train_gene_names, self.gene2id, self.subclass2id, self.args, subsample_input=False, path="/scratch/baffou/pkl/10_bins/test_1_percentile.pkl")
        self.test2 =       SingleDataset(     test2, self.train_gene_names, self.gene2id, self.subclass2id, self.args, subsample_input=False, path="/scratch/baffou/pkl/10_bins/test_2_percentile.pkl")
        if self.args.eval_ood: 
            self.ood  =    SingleDataset(     ood,   self.total_gene_names, self.gene2id, self.subclass2id, self.args, subsample_input=False, path="/scratch/baffou/pkl/10_bins/ood_percentile.pkl")

        self.eval_datasets = {'test1':self.test1, 'test2':self.test2}
        if self.args.eval_ood: self.eval_datasets['hippocampus']=self.ood #, 'embryo':self.ood2}

        embeddings = []
        for name in self.total_gene_names:
            embeddings.append(loaded_embeddings[name])
        self.embeddings = torch.stack(embeddings).float()
        self.loaded_embeddings_dim = self.embeddings.size(1)
    
        self.dataloader = torch.utils.data.DataLoader(self.dataset, 
                                                      batch_size = args.batch_size, 
                                                      collate_fn=self.dataset.collate_fn, 
                                                      shuffle=True, 
                                                      num_workers=args.num_workers)

        ### MODEL

        if self.args.load_embeddings:
            self.input_dim = self.loaded_embeddings_dim
        else:
            self.input_dim = args.embed_dim
        self.embed_dim = args.embed_dim

        self.logger.info("Building model")
        if args.model =='bert':
            self.model = Bert(vocab_size=len(self.gene2id), 
                              pad_idx=self.gene2id["PAD"], 
                              input_dim = self.input_dim,
                              hidden_size=self.args.embed_dim, 
                              num_hidden_layers=self.args.n_layers, 
                              num_attention_heads=self.args.n_heads, 
                              positional_encodings=self.args.positional_encodings,
                              adjacency_path=self.args.adjacency_path,
                              bin_num=self.args.bin_num,
                              dropout=self.args.dropout,
                              max_len = self.args.max_len,
                              args=args
                              )
            if self.args.load_embeddings:
                self.model.model.embeddings.word_embeddings.weight.data[:-1].copy_(self.embeddings)
            if self.args.freeze_embeddings:
                self.model.model.embeddings.word_embeddings.weight.requires_grad = False
        elif args.model =='mlp':
            self.input_dim = self.n_genes_train
            self.model = FullyConnectedNetwork(input_size=self.input_dim,
                                               layer_sizes=[self.args.embed_dim]*self.args.n_layers,
                                               output_size=self.args.embed_dim
                                               )
            self.args.use_expression = False
    
        if self.args.loss in ["xe", "classification"]:
            n_classes = self.n_classes if self.args.loss=='classification' else self.args.n_bins+1
            self.model.classifier = nn.Linear(self.args.embed_dim, n_classes)

        self.model = self.model.to(self.device)
        if self.args.reload_checkpoint:
            weights = torch.load(Path(self.args.reload_checkpoint)/'checkpoint.pth', map_location=self.device)
            self.model.load_state_dict(weights)

        if self.args.loss == 'triplet':
            self.criterion = nn.TripletMarginLoss(margin=args.margin)
        elif self.args.loss == 'ranking':
            self.criterion = nn.MarginRankingLoss(margin=args.margin)
        elif self.args.loss in ['MAE','MRE']:
            self.criterion = nn.L1Loss()
        elif self.args.loss in ['ratio']:
            self.criterion = nn.L1Loss()
        elif self.args.loss in ['xe']:
            self.args.threshold = 0
            weight = torch.ones(self.args.n_bins+1).to(self.device)
            weight[-1]=0
            self.criterion = nn.CrossEntropyLoss(weight=weight)
        elif self.args.loss in ['classification']:
            self.criterion = nn.CrossEntropyLoss()
        args.optimizer = args.optimizer+',lr={:f}'.format(args.lr)
        self.optimizer = get_optimizer(self.model.parameters(), args.optimizer)

    def train_contrastive(self, device):

        self.model.train()
        self.model = self.model.to(device)

        losses, accs = [], []
        tmp_loss, tmp_acc = [],[]
        for i, batch in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            if self.args.model in ["sparse", "bert"]:
                anchor, expression_a, positive, expression_p, dist_p, negative, expression_n, dist_n = batch
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                expression_a, expression_p, expression_n = expression_a.to(device), expression_p.to(device), expression_n.to(device)
                if self.args.use_expression:
                    a, p, n = self.model(anchor, expression=expression_a), self.model(positive, expression=expression_p), self.model(negative, expression=expression_n)
                else:
                    a, p, n = self.model(anchor), self.model(positive), self.model(negative)
            elif self.args.model in ["dense","mlp"]:
                anchor, positive, dist_p, negative, dist_n = batch
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                a, p, n = self.model(anchor), self.model(positive), self.model(negative)

            dist_p, dist_n = dist_p.to(device), dist_n.to(device)
            if self.args.loss == 'triplet':
                loss = self.criterion(a, p, n)
            elif self.args.loss == 'ranking':
                loss = self.criterion(a, p, n)
            elif self.args.loss == 'ratio':
                pred_p, pred_n = (a-p).norm(dim=-1), (a-n).norm(dim=-1)
                true_ratio = dist_p/dist_n
                pred_ratio = pred_p/(pred_n+1.e-12)
                loss = self.criterion(pred_ratio, true_ratio.squeeze(1))
            elif self.args.loss == 'MAE':
                loss_p = self.criterion((a-p).norm(dim=-1), dist_p.squeeze())
                loss_n = self.criterion((a-n).norm(dim=-1), dist_n.squeeze())
                loss = loss_p+loss_n
            elif self.args.loss == 'MRE':
                avg_dist = .5*(dist_p.mean()+dist_n.mean())
                dist_p, dist_n = dist_p.to(device), dist_n.to(device)
                loss_p = self.criterion((a-p).norm(dim=-1), dist_p.squeeze())
                loss_n = self.criterion((a-n).norm(dim=-1), dist_n.squeeze())
                loss = (loss_p*dist_n.mean()+loss_n*dist_p.mean())/dist_p.mean()
            elif self.args.loss == 'MARE':
                pred_p, pred_n = (a-p).norm(dim=-1), (a-n).norm(dim=-1)                
                dist_p, dist_n = dist_p.to(device), dist_n.to(device)
                loss_p = (torch.abs(pred_p-dist_p)/dist_p).clamp(0,100).mean()
                loss_n = (torch.abs(pred_n-dist_n)/dist_n).clamp(0,100).mean()
                loss = (loss_p + loss_n)*dist_p.mean()
            elif self.args.loss == 'xe':
                #pred_n = torch.cat((a,n), dim=-1)
                pred_n = a-n
                pred_n = self.model.classifier(pred_n)
                dist_n = bin_tensor(dist_n.squeeze(1), k=self.args.n_bins)       
                loss_n = self.criterion(pred_n, dist_n)
                loss = loss_n                
                # pred_p = torch.cat((a,p), dim=-1)
                # pred_p = self.classifier(pred_p)
                # dist_p = bin_tensor(dist_p.squeeze(1), k=self.args.n_bins) 
                # loss_p = self.criterion(pred_p, dist_p)
                # loss += loss_p
            else:
                raise NotImplementedError
            
            if self.args.loss != 'xe':
                acc = ((a-p).norm(dim=-1) < (a-n).norm(dim=-1)).sum()/a.size(0)
            else:
                acc = ((pred_n.argmax(dim=-1) == dist_n)).sum()/a.size(0)
            tmp_loss.append(loss.item()), tmp_acc.append(acc.item())

            if i%self.args.print_freq==0: 
                self.logger.info("TRAIN batch {0:d}\t{{'loss': {1:.6f}, 'acc': {2:.6f}, 'lr': {3:.6f}, 'max mem': {4:.2f}}}".format(i, np.mean(tmp_loss), np.mean(tmp_acc), self.optimizer.param_groups[0]["lr"], torch.cuda.max_memory_allocated()/1024**3))
                
                losses.extend(tmp_loss)
                accs.extend(tmp_acc)
                tmp_loss, tmp_acc = [],[]
                        
            loss.backward()
            self.optimizer.step()

            if i==self.args.eval_freq:
                break 

        return np.mean(losses), np.mean(accs)

    def eval_contrastive(self, name, device, dataset, n=None):

        self.model.eval()
        self.model = self.model.to(device)
        
        if n is None:
            n = len(dataset)
        else:
            n = min(n, len(dataset))
            dataset = copy.deepcopy(dataset)
            dataset.data = dataset.data[:n]

        points = np.zeros((n,2))
        for i, (_, cell) in enumerate(dataset.data.iterrows()):
            points[i][0] = cell['coord_X']
            points[i][1] = cell['coord_Y']

        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size = self.args.batch_size_eval, 
                                                 collate_fn= self.dataset.collate_fn, 
                                                 shuffle = False, 
                                                 num_workers = self.args.num_workers)

        pred_distances = np.zeros((n,n))
        true_distances = np.zeros((n,n))
        
        embeddings = []
        indices, expressions = [],[]
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if self.args.model in ["bert"]:
                    indices, expressions, subclass = batch
                    indices, expressions, subclass = indices.to(device), expressions.to(device), subclass.to(device)
                    if self.args.use_expression:
                        embedding = self.model(indices, expression = expressions)
                    else:
                        embedding = self.model(indices)
                elif self.args.model in ["mlp"]:
                    expressions = batch[0]
                    expressions = expressions.to(device)
                    embedding = self.model(expressions)
                embeddings.append(embedding)
        embeddings = torch.cat(embeddings)

        for i, point in enumerate(points):
            distances = (point[0]-points[:,0])**2 + (point[1]-points[:,1])**2    
            true_distances[i,:]=distances
            embedding = embeddings[i]
            if self.args.loss != 'xe':
                pred_distances[i,:] = (embedding-embeddings).norm(dim=-1).cpu().detach().numpy()
            else:
                pred_distances[i,:] = self.model.classifier(embedding-embeddings).argmax(dim=-1).cpu().detach().numpy()

        precisions, auc_scores = [], []
        for percentile in self.args.percentiles.split(','):
            percentile = float(percentile)
            true_contacts = true_distances<np.quantile(true_distances, percentile)
            pred_contacts = pred_distances<np.quantile(pred_distances, percentile)
            predictions = pred_contacts.flatten()
            scores = -pred_distances.flatten()/pred_distances.max()+1
            labels = true_contacts.flatten()
            precision = precision_score(labels, predictions)
            # recall = recall_score(labels, predictions)
            auc_score = roc_auc_score(labels, scores)
            precisions.append(precision)
            auc_scores.append(auc_score)

        self.logger.info(f"EVAL \t{{'name':'{name}', 'precision': {precisions}, 'auc': {auc_scores}}}")

        return pred_distances, true_distances
 
    def train_classification(self, device):

        self.model.train()
        self.model = self.model.to(device)

        losses, accs = [], []
        tmp_loss, tmp_acc = [],[]
        for i, batch in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            anchor, expression, subclass = batch
            anchor, expression, subclass = anchor.to(device), expression.to(device), subclass.to(device).squeeze(-1)
            if self.args.use_expression:
                pred = self.model(anchor, expression=expression)
            else:
                pred = self.model(anchor)
            pred = self.model.classifier(pred)
 
            loss = self.criterion(pred, subclass.squeeze(-1))
            acc = ((pred.argmax(dim=-1) == subclass)).sum()/pred.size(0)
            tmp_loss.append(loss.item()), tmp_acc.append(acc.item())

            if i%self.args.print_freq==0: 
                self.logger.info("TRAIN batch {0:d}\t{{'loss': {1:.6f}, 'acc': {2:.6f}, 'lr': {3:.6f}, 'max mem': {4:.2f}}}".format(i, np.mean(tmp_loss), np.mean(tmp_acc), self.optimizer.param_groups[0]["lr"], torch.cuda.max_memory_allocated()/1024**3))
                losses.extend(tmp_loss)
                accs.extend(tmp_acc)
                tmp_loss, tmp_acc = [],[]
                        
            loss.backward()
            self.optimizer.step()

            if i==self.args.eval_freq:
                break 

        return np.mean(losses), np.mean(accs)
    
    def eval_classification(self, name, device, dataset):

        self.model.eval()
        self.model = self.model.to(device)

        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size = self.args.batch_size_eval, 
                                                 collate_fn= self.dataset.collate_fn, 
                                                 shuffle = False, 
                                                 num_workers = self.args.num_workers)

        losses, accs = [], []
        for i, batch in enumerate(dataloader):
            anchor, expression, subclass = batch
            anchor, expression, subclass = anchor.to(device), expression.to(device), subclass.to(device).squeeze(-1)

            if self.args.use_expression:
                pred = self.model(anchor, expression=expression)
            else:
                pred = self.model(anchor)
            pred = self.model.classifier(pred)
            
            loss = self.criterion(pred, subclass.squeeze(-1))
            acc = ((pred.argmax(dim=-1) == subclass)).sum()/pred.size(0)
            losses.append(loss.item()), accs.append(acc.item())
                        
        self.logger.info("EVAL \t{{'name':'{0}', 'loss': {1:.6f}, 'acc': {2:.6f}, 'max mem': {3:.2f}}}".format(name, np.mean(losses), np.mean(accs), torch.cuda.max_memory_allocated()/1024**3))

        return np.mean(losses), np.mean(accs)
    
    def train_one_epoch(self, epoch):

        if self.args.loss == 'classification':
            loss, acc = self.train_classification(self.device)
        else:
            loss, acc = self.train_contrastive(self.device)
        self.logger.info(f"TRAIN epoch {epoch}: {{'loss': {loss}, 'acc': {acc}}}")

    def evaluate(self):
        
        if self.args.loss == 'classification':
            eval_fn = self.eval_classification
            kwargs = {}
        else:
            eval_fn = self.eval_contrastive
            kwargs = {'n': 1000}

        for name, dataset in self.eval_datasets.items():
            self.logger.info(f"Running eval on {name}")
            if name=="hippocampus":
                for n in [1,2,4]:
                    self.model.max_len = n*len(self.train_gene_names)
                    new_name = f"{name}_{n}"
                    eval_fn(new_name, self.device, dataset, **kwargs)
            else:
                self.model.max_len = None
                eval_fn(name, self.device, dataset, **kwargs)

    def train(self):

        if self.args.eval_only:
            self.evaluate()
            exit(0)

        for epoch in range(self.args.n_epochs):
            self.train_one_epoch(epoch)
            self.evaluate()
            torch.save(self.model.state_dict(), self.dump_path / 'checkpoint.pth')

def get_parser():
    parser = argparse.ArgumentParser(description="Pairwise distance between cell embeddings.", add_help=False)

    # GENERAL
    parser.add_argument("--seed", type=int, default=0, help="manual seed")
    parser.add_argument("--device", type=str, default="cuda", help="device to run on")
    parser.add_argument("--gpu_index", type=str, default=None, help="device index to run on")
    parser.add_argument("--exp_name", type=str, default='raw_no_prior_percentile', help="path to save logs and weights")
    parser.add_argument("--process_name", type=str, default=None, help="process name (default exp_name)")
    parser.add_argument("--exp_folder", type=str, default='/scratch/baffou/experiments/no_binning/percentile', help="path to save logs and weights")
    parser.add_argument("--data_path", type=str, default="/data1/mlbiodata1/baffou/data/cell2loc/", help="data path")
    parser.add_argument("--reload_checkpoint", type=str, default=None, help="reload checkpoint path")
    parser.add_argument("--eval_only", type=bool_flag, default=False, help="eval only")
        
    # LOGGING
    parser.add_argument("--print_freq", type=int, default=10, help="Print loss every n batches.")
    parser.add_argument("--eval_freq", type=int, default=1000, help="Eval model every n batches.")
    parser.add_argument("--percentiles", type=str, default='0.01,0.1', help="Test precision on various percentiles.")
    parser.add_argument("--eval_ood", type=bool_flag, default=True, help="Eval model on ood data.")

    # MODEL
    parser.add_argument("--model", type=str, default="bert", choices=["sparse","dense","bert","mlp"], help="Input dimension for the Transformer model.")
    parser.add_argument("--dropout", type=float, default=0., help="Number of attention heads.")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of encoder layers.")
    parser.add_argument("--max_len", type=int, default=1000, help="Number of encoder layers.")

    # HPARAMS
    parser.add_argument("--optimizer", type=str, default="adam_cosine,warmup_updates=1000,init_period=10000,period_mult=1.5,lr_shrink=0.5", help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading. The more the better")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--batch_size_eval", type=int, default=8, help="Batch size eval.")
    parser.add_argument("--n_epochs", type=int, default=1000, help="Number of training epochs.")

    # TASK
    parser.add_argument("--loss", type=str, default='triplet', help="loss function to use")
    parser.add_argument("--n_bins", type=int, default=20, help="number of bins for xe loss")
    parser.add_argument("--normalize_expression", type=bool_flag, default=True, help="whether to use log normalized expression values")
    parser.add_argument("--use_expression", type=bool_flag, default=True, help="whether to use expression values")
    parser.add_argument("--expression_as_input", type=bool_flag, default=True, help="whether to use expression values as input rather than mult attention")
    parser.add_argument("--sparse_input", type=bool_flag, default=False, help="whether to keep only nonzero expression")
    parser.add_argument("--subsample_input", type=bool_flag, default=True, help="whether to sample random number of genes")
    parser.add_argument("--load_embeddings", type=bool_flag, default=True, help="whether to load embeddings")
    parser.add_argument("--freeze_embeddings", type=bool_flag, default=True, help="whether to freeze loaded embeddings")
    parser.add_argument("--num_neighbors", type=int, default=10, help="Number of neighbors to choose.")
    parser.add_argument("--negative_dist", type=int, default=3, help="Number of neighbors to choose.")
    parser.add_argument("--same_subclass_prob", type=float, default=0, help="Percentage of samples where negatives come from same cell subclass.")
    parser.add_argument("--threshold", type=float, default=0.0005, help="Distance threshold of neighborhood.")
    parser.add_argument("--margin", type=float, default=1., help="Margin for the Contrastive Loss function.")
    parser.add_argument("--positional_encodings", type=str, default=None, help="Type of positional encoding to use.")
    parser.add_argument("--adjacency_path", type=str, default=None, help="Path to the adjacency matrix used for positionnal encodings.") 
    parser.add_argument("--selected_genes_path", type=str, default=None, help="Path to the list of selected genes.")
    parser.add_argument("--bin_num", type=int, default=0, help="Number of bins. If 0, then keep raw expressions.")  

    
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    trainer = Trainer(args)
    print("NUMBER OF GENES IN TRAINING", trainer.n_genes_train)
    print('NUMBER OF GENES IN OOD', len(trainer.ood_gene_names))
    print("NUMBER OF GENES IN TOTAL", trainer.n_genes_total)
    trainer.train()

if __name__ == "__main__":
    main()
