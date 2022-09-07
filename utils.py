import yaml
from yaml.loader import FullLoader
from torch import tensor,LongTensor,long,nn,optim,argmax
from numpy import nan
from tqdm.auto import tqdm 

class configurations():
    def __init__(self,file: str = './conf/config.yaml'):
        self.file = file
        
    def load_config(self):
        stream = open(self.file,'r')
        self.config = yaml.load(stream,Loader=FullLoader)

    def update_config(self):
        with open(self.file,'w') as f:
            f.write(yaml.dump(self.config))

class LossFn(nn.Module):
    def __init__(self,weight=None):
        super(LossFn,self).__init__()
        self.ls_fn = nn.CrossEntropyLoss()
        
    def forward(self,out,targ):
        return self.ls_fn(out,targ)


class DataSet():
    def __init__(self,inputs,labels,tokenizer,max_length):
        self.data_x = inputs
        self.labels = LongTensor(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __getitem__(self,idx):
    
        tokens = self.tokenizer(self.data_x[idx],max_length=self.max_length,truncation=True,padding=True,add_special_tokens=True)
        ids = tokens['input_ids']
        token_type_id = tokens['token_type_ids']
        mask = tokens['attention_mask']
    
        padding_len = self.max_length-len(ids)
        ids = ids+([0]*padding_len)
        token_type_id = token_type_id + ([0]*padding_len)
        mask = mask + ([0]*padding_len)
        
        
        
        items = {'input_ids':tensor(ids,dtype=long),
                 'token_type_ids':tensor(token_type_id,dtype=long),
                 'attention_mask':tensor(mask,dtype=long)}
        items['label'] = self.labels[idx]
        
        return (items)

    def __len__(self):
        return len(self.labels)



class Analyzer:
    def __init__(self,model,tokenizer):
        self.model = model
        self.tokenizer = tokenizer





def train(data_loader,model,criterion,optimizer,device,scheduler,epoch):
    model.train()
    data_len = len(data_loader.dataset)

    for i in range(epoch):
        for idx, batch in tqdm(enumerate(data_loader),desc="Train Batches: "):

            input_id = batch['input_ids']
            token_type_id = batch['token_type_ids']
            mask = batch['attention_mask']
            label = batch['label']
            
            optimizer.zero_grad()

            out = model(input_ids=input_id,token_type_ids=token_type_id,attention_mask=mask)
            loss = criterion(out,label)
            loss.backward()

            optimizer.step()
            scheduler.step()

            if idx % 100 == 0 or idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, idx * len(input_id), data_len,
                    100 * idx / len(data_loader), loss.item()))


def eval(data_loader,model,criterion):
    model.eval()
    test_loss = []
    outputs = []
    labels = []
    for batch in data_loader:

        input_id = batch['input_ids']
        token_type_id = batch['token_type_ids']
        mask = batch['attention_mask']
        label = batch['label']
        out = model(input_ids=input_id,token_type_ids=token_type_id,attention_mask=mask)
        loss = criterion(out,label)

        test_loss.append(loss.item())
        outputs.append(list(out.argmax(dim=-1).data.numpy()))
        labels.append(list(label.data.numpy()))
    
    return sum(test_loss)/len(data_loader.dataset),outputs,labels

    



def predictions(text,model,tokenizer,max_length):
    
    tokens = tokenizer(text,max_length=max_length,truncation=True,padding=True,add_special_tokens=True)
    ids = tokens['input_ids']
    token_type_id = tokens['token_type_ids']
    mask = tokens['attention_mask']

    padding_len = max_length-len(ids)
    ids = ids+([0]*padding_len)
    token_type_id = token_type_id + ([0]*padding_len)
    mask = mask + ([0]*padding_len)

    input_ids = tensor(ids,dtype=long),
    token_type_ids = tensor(token_type_id,dtype=long),
    attention_mask = tensor(mask,dtype=long)

    prediction = argmax(model(input_ids = input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)).data.detach().numpy()[0]

    return prediction


