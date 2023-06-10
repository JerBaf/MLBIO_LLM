import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

class ContrastiveDataset(Dataset):
    def __init__(self, df, gene_names, gene2id, args, subsample_input=None, path=''):

        self.path = path
        self.args = args
        self.bin_num = args.bin_num
        self.threshold = args.threshold
        self.data = df
        self.gene_names = gene_names
        self.gene2id = gene2id
        self.preprocess_data()
        if subsample_input is None:
            self.subsample_input = args.subsample_input
        else:
            self.subsample_input = subsample_input

    def bin_expr(self,expression:float,min_expr:float,max_expr:float,bin_num:int):
        """ Min-max scale the expression before binning it in the interval [0,bin_num-1]. """
        min_max_scaled = (expression - min_expr)/(max_expr - min_expr)
        expr_vector = np.zeros(bin_num,dtype=np.float32)
        bin_id = int(min_max_scaled*bin_num)
        if bin_id == bin_num:
            bin_id = bin_id -1
        expr_vector[bin_id] = 1.0
        return expr_vector
    
    def bin_expr_percentile(self,expression:float,steps:list,bin_num:int):
        """ Bin the expression into a vector using a list of percentiles. """
        expr_vector = np.zeros(bin_num,dtype=np.float32)
        for i in range(bin_num):
            if steps[i] >= expression:
                expr_vector[i] = 1.0
                return expr_vector


    def preprocess_data(self):
        if self.args.normalize_expression:
            # can't do this because forces unexpressed genes to be expressed
            #for key in self.gene_names: 
            #    self.data[key] = np.log2(1+self.data[key])
            #    if self.data[key].mean()!=0:
            #        self.data[key] = self.data[key]/self.data[key].mean()
            # can't do this because adding more genes reduces the expression values
            #self.data[self.gene_names] = self.data[self.gene_names].apply(lambda x: np.log2(1 + 1e4 * x/x.sum()), axis=1)
            self.data[self.gene_names] = self.data[self.gene_names].copy().apply(lambda x: np.log2(1 + x))
            # Apply binning if specified
            if self.bin_num > 0:
                try:
                    print('DEBUG, Try Loading binned expressions.')
                    with open(self.path, 'rb') as f:
                            binned_expr = pickle.load(f)
                    self.data[self.gene_names] = pd.DataFrame(binned_expr)
                except:
                    print('DEBUG, Error while loading, construct from scratch.')
                    binned_expr = []
                    for _, row_data in self.data.iterrows():
                        min_expr = row_data[self.gene_names].min()
                        max_expr = row_data[self.gene_names].max()
                        steps = []
                        for i in range(1,self.bin_num+1):
                            steps.append(np.percentile(row_data[self.gene_names],100*i/self.bin_num))
                        #processed_expr = row_data[self.gene_names].copy().apply(
                        #    lambda expr: self.bin_expr(expr,min_expr,max_expr,self.bin_num))
                        processed_expr = row_data[self.gene_names].copy().apply(
                            lambda expr: self.bin_expr_percentile(expr,steps,self.bin_num))
                        binned_expr.append(processed_expr)
                    self.data[self.gene_names] = pd.DataFrame(binned_expr)
                    if self.path != '':
                        print('DEBUG, Saving Data')
                        with open(self.path, 'wb') as f:
                            pickle.dump(binned_expr, f)

        
        for key in ['coord_X', 'coord_Y']:
            if "region" in self.data.columns:
                groups = self.data.groupby("region")[key]
                min_, max_ = groups.transform("min"), groups.transform("max")
            else:
                min_, max_ = self.data[key].min(), self.data[key].max()
            self.data[key] = (self.data[key]-min_)/(max_ - min_)-.5
        return
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        df = self.data
        anchor = df.iloc[index]
        if 'region' in df.columns:
            region = anchor['region']
            #df = df.drop(df.index[index])
            region_df = df.query(f"region == '{region}'").copy()
        else:
            region_df = df
        subclass = anchor["subclass"]
       
        x,y = anchor['coord_X'], anchor['coord_Y']
        region_df['dist'] = (region_df['coord_X']-x)**2 + (region_df['coord_Y']-y)**2
        
        # region_df = region_df.sort_values("dist")
        # num_neighbors = self.args.num_neighbors
        # positives = region_df[:num_neighbors]
        # negatives = region_df[num_neighbors:num_neighbors*(self.args.negative_dist+1)].sample(n=num_neighbors,replace=False)

        positives = region_df.query(f"0 < dist < {self.threshold}")
        num_neighbors = len(positives)
        if num_neighbors == 0:
            positives = region_df.sort_values("dist")[:1]
            num_neighbors = 1
        negatives = region_df.query(f"dist > {self.threshold}")
        negatives = region_df.sort_values("dist")
        if np.random.rand() < self.args.same_subclass_prob:
           negatives_ = negatives.query(f"subclass == '{subclass}'").copy()
           if len(negatives_) < num_neighbors: pass
           else: negatives = negatives_
        negatives = negatives.sample(n=num_neighbors)
                
        dist_p = torch.tensor(positives["dist"].values.astype(float)).float().unsqueeze(1)
        dist_n = torch.tensor(negatives["dist"].values.astype(float)).float().unsqueeze(1)
 
        anchor = torch.tensor(anchor[self.gene_names].values.astype(float)).float()
        anchors = anchor.unsqueeze(0).repeat(num_neighbors,1)
        positives = torch.tensor(positives[self.gene_names].values.astype(float)).float()
        negatives = torch.tensor(negatives[self.gene_names].values.astype(float)).float()

        if self.args.model in ["mlp"]:
            return anchors, positives, dist_p, negatives, dist_n
        if self.args.model in ["bert"]:
            anchors, expression_a   = self.select_genes(anchors  )
            positives, expression_p = self.select_genes(positives)
            negatives, expression_n = self.select_genes(negatives)
            return anchors, expression_a, positives, expression_p, dist_p, negatives, expression_n, dist_n
        
    def get_anchor(self, index, df):
        anchor = df.iloc[index]
        anchor = torch.tensor(anchor[self.gene_names].values.astype(float)).float()
        if self.args.model in ["sparse","bert"]:
            indices, expressions = self.select_genes([anchor])
            return indices[0], expressions[0]
        else:
            return anchor

    def collate_fn(self, elements):
        res = []
        for x in zip(*elements):
            if self.args.model in ['dense',"mlp"]:
                res.append(torch.cat(x,dim=0))
            elif self.args.model in ["sparse","bert"]:
                res.append(self.pad_sequences(x))
        return tuple(res)
    
    def select_genes(self, sequences):
        indices, expressions = [], []
        for seq in sequences:
            if self.args.sparse_input:
                idx = seq.nonzero(as_tuple=True)[0]
            elif self.subsample_input:
                # sample the number of genes to use
                num_genes = np.random.randint(int(len(seq)//2), len(seq))
                idx = np.random.choice(len(seq), num_genes, replace=False)
                idx = torch.LongTensor(idx)
            else:
                idx = torch.arange(len(seq))
            exp = seq[idx]
            idx = torch.LongTensor([self.gene2id[self.gene_names[i]] for i in idx])
            indices.append(idx)
            expressions.append(exp)
        return indices, expressions

    def pad_sequences(self, sequences):
        tmp = []
        for lst in sequences:
            tmp.extend(lst)
        sequences = tmp
        lengths = torch.LongTensor([len(s) for s in sequences])

        if isinstance(sequences[0], torch.LongTensor):
            sent = torch.LongTensor(lengths.size(0), lengths.max().item()).fill_(self.gene2id["PAD"])
        else:
            if self.args.bin_num == 0:
                sent = torch.Tensor(lengths.size(0), lengths.max().item()).fill_(0)
            else:
                sent = torch.Tensor(np.zeros((lengths.size(0), lengths.max().item(), self.bin_num)))

        #print(f'DEBUG, type(sequences):{type(sequences)}, len(sequences):{len(sequences)}')
        for i, s in enumerate(sequences):
            sent[i, :lengths[i]].copy_(s)

        return sent

    def build_ood(self, area = 'mousehippocampus', restrict_gene_names=True):
        csv_path = self.args.data_path+area+'_expression.csv'
        df = pd.read_csv(csv_path, index_col=0)
        df = df.transpose()

        if restrict_gene_names:
            df = df[self.gene_names]
        #     gene_names = dataset.gene_names
        #     new = []
        #     for gene_name in gene_names:
        #         if gene_name in df.columns:
        #             new.append(gene_name)
        #     gene_names = new

        csv_path = self.args.data_path+area+'_spatial.csv'
        df2 = pd.read_csv(csv_path, index_col=0)
        for col in df2:
            df[col] = df2[col]

        df = df.rename(columns={'X':'coord_X', 'Y':'coord_Y', 'cell_type':'subclass'})

        df.to_csv(f'{self.args.data_path}/ood_{area}.csv')

class SingleDataset(ContrastiveDataset):
    def __init__(self, df, gene_names, gene2id, subclass2id, args, subsample_input=None, path=''):
        super().__init__(df, gene_names, gene2id, args, subsample_input, path=path)
        self.subclass2id = subclass2id

    def __getitem__(self, index):
        df = self.data
        anchor = df.iloc[index]
        subclass = anchor['subclass']
        if self.args.loss == "classification":
            subclass = torch.LongTensor([self.subclass2id[subclass]]).unsqueeze(0)
        else:
            subclass = torch.LongTensor([0]).unsqueeze(0)
        
        # Get Anchor post Binning
        if self.args.bin_num == 0:
            anchor = torch.tensor(anchor[self.gene_names].values.astype(float)).float()
        else:
            anchor = anchor[self.gene_names].values
            #print(f'DEBUG, Type of anchor: {type(anchor)}, shape: {anchor.shape}')
            #print(f'DEBUG, First five entries: {anchor[:5]}, type: {type(anchor[0])}, len: {len(anchor[0])}')
            new_anchor = np.zeros((anchor.shape[0],self.args.bin_num))
            for i in range(anchor.shape[0]):
                for j in range(anchor[0].shape[0]):
                        new_anchor[i,j] = anchor[i][j]
            anchor = torch.tensor(new_anchor).float()
            del new_anchor

        if self.args.model in ["bert"]:
            indices, expressions = self.select_genes([anchor])
            return indices, expressions, subclass
        else:
            return anchor.unsqueeze(0), anchor.unsqueeze(0), subclass
        
class DoubleDataset(ContrastiveDataset):
    def __init__(self, df, gene_names, gene2id, args, subsample_input=None):
        super().__init__(df, gene_names, gene2id, args, subsample_input)

    def __getitem__(self, index):
        df = self.data
        anchor = df.iloc[index]
        anchor = torch.tensor(anchor[self.gene_names].values.astype(float)).float()
        other = df.sample(n=1)
        other = torch.tensor(other[self.gene_names].values.astype(float)).float()
        dist = (anchor['coord_X'] - other['coord_X'])**2 + (anchor['coord_Y'] - other['coord_Y'])**2
        if self.args.model in ["sparse","bert"]:
            i1, e1 = self.select_genes([anchor])
            i2, e2 = self.select_genes([other])
            return i1, e1, i2, e2, dist
        else:
            return anchor.unsqueeze(0), other.unsqueeze(0)