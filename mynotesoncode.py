import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

spacy_ger = spacy.load('de') #german tokenizer
spacy_eng = spacy.load('en') #english tokenizer

def tokenizer_ger(text):
return [tok.text for tok in spacy_ger.tokenizer(text)] # converts german text to tokens(eg: 'Hello my name' -> ['Hello','my','name']

def tokenizer_ger(text):
return [tok.text for tok in spacy_ger.tokenizer(text)] # converts text to tokens(eg: 'Hello my name' -> ['Hello','my','name']

german = Field(tokenize = tokenizer_ger, lower=True, init_token='<sos>', eos_token='<eos>')
english = Field(tokenize=tokenizer_eng, lower=True, init_token='<sos>', eos_token='<eos>')

train_data, validation_data, test_data = Multi30k.splits(exts=('.de','.en'),fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)   #min_freq:number of times word must repeat in dataset to be considered.
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Encoder(nn.Module):
def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
super(Encoder, self).__init__()
self.hidden_size = hidden_size
self.num_layers = num_layers

self.dropout = nn.Dropout(p)
self.embedding = nn.Embedding(input_size, embedding_size) #embedding size is output
self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, p) #self.rnn accepts self.embedding's output.hidden_size represents self.rnn output

def forward(self,x):     ## x = sentence which is tokenized and mapped to index depending on its location in the vocab and that vector will be sent to lstm, x.shape:(seq_length,N) where N is the batch size.
embedding = self.dropout(self.embedding(x))
#embedding shape:(seq_length, N, embedding_size): basically each word is mapped to a dimensional space of embedding size which can be for eg 300

outputs, (hidden,cell) = self.rnn(embedding)

return hidden, cell

class Decoder(nn.Module):
def __init__(self, input_size, embedding_size, hidden_sizem
output_size, num_layers, dropout): #output size: if vocab size is 10000 then output size represents  the probability that word of input is either one of those 10000 words.output size is same as input size
super(Decoder, self).__init__()
self.hidden_size = hidden_size
self.num_layers = num_layers

self.dropout = nn.Dropout(p)
self.embedding = nn.Embedding(input_size, embedding_size) #embedding size is output
self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, p) 
self.fc = nn.Linear(hidden_size, output_size) #output_size is same size as vocabulary

def forward(self, x, hidden, cell):
#shape if x: we get (N) but we want (1,N) as 1 word is predicted, unlike in encoder where an entire sentence is taken hence x equals seq length,N
x = x.unsqueeze(0)

embedding = self.dropout(self.embedding(x))
# embedding shape: (1, N, embedding_size)

outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell)) #output will be the word that is predicted and hidden and cell will be the variables which will be involved in the prediction
#shape of outputs: (1,N, hidden_size)

self.fc(outputs)  
#shape of predictions: (1, N, length_of_vocab)

predictions = predictions.squeeze(0) # to remove 1 from above vector as that is how pytorch will accept the vectors furthermore that is the arrangement which will be required to predict the full sentence later on

return predictions, hidden, cell

class Seq2Seq(nn.Module):   #combines encoder and decoder
def __init__(self, encoder, decoder):
super(Seq2Seq, self).__init__()
self.encoder = encoder
self.decoder = decoder

#watch 20 min onwards of this video Pytorch Seq2Seq Tutorial for Machine Translation for teacher_force_ratio explanation
def forward(self, source, target, teacher, teacher_force_ratio=0.5):
batch_size= source.shape[1] ## sourceshape is [target_len, N]
target_len = target.shape[0]
target_vocab_size = len(english.vocab)
outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device) # each word predicted in target len will be so for an entire batch and every prediction will be a vector of an entire vocab the output tensor in iteration below will be batch_size,target_vocab_size in shape
hidden, cell = self.encoder(source)  #hidden and cell are constructed to be sent to the decoder for predictions

# Grab start tokenize
x = target[0] 

for t in range(1, target_len): # send word to the decoder one by one
output, hidden, cell = self.decoder(x, hidden, cell) # hidden and cell used inintially are from encoder, then the output derived from it will be used as the next hidden and cell during iteration
outputs[t]=output # we are adding the new values along the first dimension of the vector of output above which is (target_len, batch_size, target_vocab_size).to(device)
# the input of outputs above the vector (batch_sie, target_vocab_size) so we want the argmax of the second dimension to determine the predicted word.
best_guess = output.argmax(1)
x = target[t] if random.random() <  #next input to the decoder will be the target word if random.random() (which will be between zero and 1) is less than teacher force ratio

return outputs

# Training hyperparameters
num_epochs =20
learning_rate=0.001
batch_size = 64

#Model hyperparameters
load_model = False
device = torch.device()
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
mun_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

#Tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
(train_data, validation_data, test_data),
batch_size= batch_sizesort_within_batch= True,
sort_within_batch = True,
sort_key = lambda x: len(x.src), device=device) #bucket iterator selects batches in which the source and target len is similar so less padding and therefore less computing is utilized

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate
pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)) #ignore_index prevents the registering of the padding portion in the loss function

if load_model:
  load_checkpoint(torch.load('my_checkpoint.pth.ptar'), model, optimizer)
  
  for epoch in range(num_epochs):
  print(f'Epoch [{epoch}/{num_epochs}]')
  
  checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
  save_checkpoint(checkpoint)
  model.eval()
  translated_sentence = translate_sentence(model, sentence, german, english,device, max_length=50)
  
  print(f'Translated example sentence \n {translated_sentence}')
  for batch_idx, batch in enumerate(train_iterator):
  inp_data = batch.src.to(device)
  target = batch.trg.to(device)
  output = model(inp_data, target)
  #output shape: (trg_len, batch_size, output_dim) we need to eliminate the first dimension as crossentropy only accepts 2 dimensional vectors so we will multiply trg_len with batch_size
  output= output[1L].reshape(-1, output.shape[2]) # in order to remove start token we start from the second output hence [1:]
  target = target[1:].reshape(-1_
  optimizer.zero_grad()
  loss = criterion(output, target)
  loss.backward()
  
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
  
  optimizer.step()
  
  writer.add_scalar('Training loss', loss, global_step=step)
  step += 1
  