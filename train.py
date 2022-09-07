import hydra
from torch.utils.data import DataLoader
from utils import *
import pandas as pd
import os
from BertModel import *
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers.optimization import get_linear_schedule_with_warmup


@hydra.main(config_path='./conf',config_name='config')
def main(cfg):
    df_path = os.path.join(cfg.paths.manifest_path,cfg.manifest_files.data_name)

    df = pd.read_csv(df_path).drop("Unnamed: 0",axis= 1).dropna()

    df = df.copy()
    df['label'] = df['sentiments'].apply(lambda x: 0 if x=='Neutral' else 1 if x=="Negative" else 2 if x == 'Positive' else None)
    df = df.drop('sentiments',axis=1)
    
    train_df,test_df = train_test_split(df,test_size=0.1,random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    train_dataset = DataSet(list(train_df['words'].values),train_df['sentiments'].values)
    test_dataset = DataSet(list(test_df['words'].values),test_df['sentiments'].values)

    train_loader = DataLoader(train_dataset,batch_size=cfg.params.batch_size,shuffle=True)
    
    if cfg.paths.tokenizer_path is None:
        try:

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            print("loaded tokenizer")

        except Exception as e:
            print(e)
    else:
        try:

            tokenizer = BertTokenizer.from_pretrained(cfg.paths.tokenizer_path)
            print("loaded tokenizer")

        except Exception as e:
            print(e)

    model = bert_model(n_out=cfg.bert_configs.n_out,path_to_model=cfg.path.model_path,model_name=cfg.bert_configs.model_name)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),lr=cfg.parmas.lr)

    total_steps = int(len(train_dataset) / 16)
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps - warmup_steps
    )

    train(train_loader,model,loss_fn,optimizer,'cpu',scheduler,10)


if __name__=="__main__":
    main()