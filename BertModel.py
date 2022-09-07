from transformers import BertModel
from transformers import BertTokenizer
import torch.nn as nn
from torch import clamp, sum
import os


class bert_model(nn.Module):
  def __init__(self,n_out:int=3,path_to_model=None,model_name=None):
    super(bert_model,self).__init__()

    if path_to_model is None:
        if not model_name:
            raise Exception(f"please provide model path, or set model_name to None")

        self.bert = BertModel.from_pretrained('bert-base-uncased')

    else:
        if model_name is None:
            raise Exception(f'please provide model name, e.g my_model.bin, or set model_path to None')

        self.bert = BertModel.from_pretrained(os.path.join(path_to_model,model_name))

    self.out = nn.Linear(768,n_out)
    self.dropout = nn.Dropout(0.1)

  def forward(self,input_ids,token_type_ids,attention_mask):
    output = self.bert(input_ids,attention_mask = attention_mask,token_type_ids=token_type_ids)
    mean_pooled = self.mean_pool(output.last_hidden_state,attention_mask)
    out = self.dropout(mean_pooled)
    return self.out(out)

  def mean_pool(self,hidden_state,mask):
    padding_expanded = mask.unsqueeze(-1).expand(hidden_state.size()).float()
    return sum(hidden_state*padding_expanded,1)/clamp(padding_expanded.sum(1),min=1e-9)

