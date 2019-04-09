
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
from tqdm import tqdm, trange

data = pd.read_csv("./data/NER/ner_dataset.csv", encoding="latin1").fillna(method="ffill")
data.tail(10)


# In[2]:


type(data)


# In[3]:


class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


# In[4]:


getter = SentenceGetter(data)


# ### This is how the sentences in the dataset look like.

# In[5]:


sentences = [" ".join([s[0] for s in sent]) for sent in getter.sentences]
sentences[0]


# ### The sentences are annotated with the BIO-schema and the labels look like this.

# In[6]:


labels = [[s[2] for s in sent] for sent in getter.sentences]
print(labels[0])


# In[7]:


tags_vals = list(set(data["Tag"].values))
tag2idx = {t: i for i, t in enumerate(tags_vals)}


# In[8]:


tag2idx


# ### Prepare the sentences and labels
# Before we can start fine-tuning the model, we have to prepare the data set for the use with pytorch and bert.

# In[9]:


import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam


# In[10]:


from pytorch_pretrained_bert.tokenization import BertTokenizer


# In[11]:


MAX_LEN = 75 ##max length of token in sequence
bs = 32  ##batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0) 


# ### The Bert implementation comes with a pretrained tokenizer and a definied vocabulary. We load the one related to the smallest pre-trained model bert-base-uncased. Try also the cased variate since it is well suited for NER.

# In[12]:


##load BertTokenizer class
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


# In[13]:


from pathlib import Path
Path.home() / '.pytorch_pretrained_bert'


# In[14]:


tokenizer


# In[15]:


### tokenize sentences

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print(tokenized_texts[0])


# In[16]:


#cut and pad the token and label sequences to our desired length.
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")


# In[17]:


tokenizer.vocab.items()


# ### The Bert model supports something called attention_mask, which is similar to the masking in keras. So here we create the mask to ignore the padded elements in the sequences.

# In[18]:


attention_masks = [[float(i>0) for i in ii] for ii in input_ids]


# In[19]:


len(attention_masks)


# In[20]:


tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, 
                                                            random_state=2018, test_size=0.1)

tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)


# Since we’re operating in pytorch, we have to convert the dataset to torch tensors.

# In[21]:


tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)


# In[22]:


input_ids[0]


# In[23]:


sentences[0]


# ### The last step is to define the dataloaders. We shuffle the data at training time with the RandomSampler and at test time we just pass them sequentially with the SequentialSampler.

# In[24]:


train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)


# In[25]:


train_dataloader


# ## Setup the Bert model for finetuning
# The pytorch-pretrained-bert package provides a BertForTokenClassification class for token-level predictions. BertForTokenClassification is a fine-tuning model that wraps BertModel and adds token-level classifier on top of the BertModel. The token-level classifier is a linear layer that takes as input the last hidden state of the sequence. We load the pre-trained bert-base-uncased model and provide the number of possible labels.

# In[26]:


#init BertForTokenClassification class

##from_pretrained from BERTFromPreTrained
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))


# Now we have to pass the model parameters to the GPU. ### why?

# In[27]:


model.cuda();


# Before we can start the fine-tuning process, we have to setup the optimizer and add the parameters it should update. A common choice is the Adam optimizer. We also add some weight_decay as regularization to the main weight matrices. If you have limited resources, you can also try to just train the linear classifier on top of Bert and keep all other weights fixed. This will still give you a good performance.

# In[28]:


FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters()) 
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)


# In[29]:


param_optimizer = list(model.classifier.named_parameters())  
### Q1:model.classifier(where comes this classifier from BertTokenClassification, but argument for classifier is different than declaration)

## model is inheritated from BERTPretrainedModel.from_pretrained, 
##classifier is self.classifier=nn.Linear in BertForTokenClassification, named_parameters is from nn.module

##because from_pretrained is a class method, so model is initialized with from_pretrained argument and
#has properties of both (BertForTokenClassification and BertFromPretrained) 


# In[30]:


model.classifier


# In[31]:


# is there any place used "forward" in BertForTokenClassification?


# In[32]:


#in tokenization.py, didn't find  declaration for cls: tokenizer = cls(resolved_vocab_file, *inputs, **kwargs)


# In[33]:


param_optimizer[1]  ## Q3 which part is pre-trained? 


# In[34]:


param_optimizer[0]


# In[35]:


model.config  ## Q4 I can't figure out where to pass this config to model


# In[36]:


#   Q5 what is this cls?
    ##def from_pretrained(cls, pretrained_model_name_or_path, state_dict=None, cache_dir=None,
    #                    from_tf=False, *inputs, **kwargs):
##cls refers as class itself


# In[37]:


#Q6: *inputs, ** kwargs, and super 
 #def __init__(self, config, *inputs, **kwargs):
 #    super(BertPreTrainedModel, self).__init__()
## super here means initialize the parent class of BertPreTRainedModel, which is nn.Module


# ### First we define some metrics, we want to track while training. We use the f1_score from the seqeval package. You can find more details here. And we use simple accuracy on a token level comparable to the accuracy in keras.

# In[38]:


from seqeval.metrics import f1_score

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# In[ ]:


epochs = 5
max_grad_norm = 1.0

for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # forward pass
        loss = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask, labels=b_labels)
        # backward pass
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        
        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss/nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))


# Evaluation

# In[ ]:


model.eval()
predictions = []
true_labels = []
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
for batch in valid_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                              attention_mask=b_input_mask, labels=b_labels)
        logits = model(b_input_ids, token_type_ids=None,
                       attention_mask=b_input_mask)
        
    logits = logits.detach().cpu().numpy()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    label_ids = b_labels.to('cpu').numpy()
    true_labels.append(label_ids)
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy

    nb_eval_examples += b_input_ids.size(0)
    nb_eval_steps += 1

pred_tags = [[tags_vals[p_i] for p_i in p] for p in predictions]
valid_tags = [[tags_vals[l_ii] for l_ii in l_i] for l in true_labels for l_i in l ]
print("Validation loss: {}".format(eval_loss/nb_eval_steps))
print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))

