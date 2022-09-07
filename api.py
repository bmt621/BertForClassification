from fastapi import FastAPI
import os
from BertModel import bert_model
from utils import configurations,predictions
from transformers import BertTokenizer
from pydantic import BaseModel


app = FastAPI()
class datas(BaseModel):
    text: str


configs = configurations(file='conf/config.yaml')
#load configs
configs.load_config()

#get model_path
model_path = configs.config.paths.model_path
model_name = configs.config.bert_configs.model_name


#get tokenizer_path
tokenizer_path = configs.config.path.tokenizer_path

if tokenizer_path is None:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncase')

else:
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

#if model_path is None, it'll download from huggingface hub,
model = bert_model(n_out=3,path_to_model=model_path,model_name=model_name)
get_map = {0:'Neutral',1:'Negetive',2:'Positive'}
@app.post('/predictions')
def prediction(sp:datas):
    predict = predictions(sp.text,model,tokenizer)
    confidence = predict
    predict = get_map[predict]

    return ({'sentiment':predict,'confidence':confidence})

