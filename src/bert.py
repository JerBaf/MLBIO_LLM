from transformers import BertModel, BertConfig
import torch.nn as nn

class Bert(nn.Module):
   def __init__(self, vocab_size, pad_idx, input_dim, hidden_size=256, num_hidden_layers=6, num_attention_heads=8,
                positional_encodings=None, adjacency_path=None, bin_num=0, dropout=0, max_len = None, args=None):
      super(Bert, self).__init__()    
       
      self.args = args
       # self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True,  output_attentions=True)
       # self.model = randomize_model(self.model)
      if max_len is None:  
         self.max_len=10000
      else:
         self.max_len=max_len
      assert hidden_size%8 == 0, "Hidden size must be multiple of number of heads"
      self.config = BertConfig(
          vocab_size= vocab_size,
          input_dim=input_dim,
          hidden_size=hidden_size, #768
          num_hidden_layers=num_hidden_layers,
          num_attention_heads=num_attention_heads, #12
          intermediate_size=4*hidden_size, #3072
          hidden_act="gelu",
          hidden_dropout_prob=dropout,
          attention_probs_dropout_prob=dropout,
          max_position_embeddings = 10000,#max_output_len,
          type_vocab_size=2,
          initializer_range=0.02,
          layer_norm_eps=1e-12,
          pad_token_id=pad_idx,
          position_embedding_type=positional_encodings,
          adjacency_path=adjacency_path,
          bin_num=bin_num,
          use_cache=True,
          classifier_dropout=None,
          output_hidden_states=True,
          output_attentions=True,
          expression_as_input = args.expression_as_input
      )
       
      self.model = BertModel(self.config)
      #del self.model.embeddings.position_ids
      #del self.model.embeddings.position_embeddings
 
   def forward(self, indices, expression=None):

      if self.max_len:
         indices, expression = indices[:,:self.max_len], expression[:,:self.max_len]

      

      outputs = self.model(indices.squeeze(1), 
                           expression=expression,
                           position_ids=None)#, z_sgs) 
      hidden_states = outputs[2][-1]
      out = hidden_states[:,-1,:]
      
      return out