import torch
import torch.nn as nn
from transformers import BertModel


class BERTGRU(nn.Module):
	def __init__(self,
				 bert,
				 hidden_dim,
				 output_dim,
				 n_layers,
				 bidirectional,
				 dropout):

		super().__init__()

		self.bert = bert

		embedding_dim = bert.config.to_dict()['hidden_size']

		self.rnn = nn.GRU(embedding_dim,
						  hidden_dim,
						  num_layers = n_layers,
						  bidirectional = bidirectional,
						  batch_first = True,
						  dropout = 0 if n_layers < 2 else dropout)

		self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

		self.dropout = nn.Dropout(dropout)

	def forward(self, text):

		#text = [batch size, sent len]

		with torch.no_grad():
			embedded = self.bert(text)[0]

		#embedded = [batch size, sent len, emb dim]

		_, hidden = self.rnn(embedded)

		#hidden = [n layers * n directions, batch size, emb dim]

		if self.rnn.bidirectional:
			hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
		else:
			hidden = self.dropout(hidden[-1,:,:])

		#hidden = [batch size, hid dim]

		output = self.out(hidden)

		#output = [batch size, out dim]

		return output, hidden, embedded

	

class BERTCNN(nn.Module):
	def __init__(self,
				 bert,
				 output_dim):

		super().__init__()

		self.bert = bert

		embedding_dim = bert.config.to_dict()['hidden_size']
		
		self.unigram = nn.Conv1d(in_channels = embedding_dim, out_channels = embedding_dim // 3, kernel_size = 1)
		
		self.trigram = nn.Conv1d(in_channels = embedding_dim, out_channels = embedding_dim // 3, kernel_size = 3, padding = 1)
		
		self.fivegram = nn.Conv1d(in_channels = embedding_dim, out_channels = embedding_dim // 3, kernel_size = 5, padding = 2)

		# self.pool = nn.MaxPool1d(embedding_dim)
		
		self.out = nn.Linear(embedding_dim, output_dim)

	def forward(self, text):

		#text = [batch size, sent len]

		with torch.no_grad():
			embedded = self.bert(text)[0]

		#embedded = [batch size, sent len, emb dim]
		
		embedded = torch.transpose(embedded, 1, 2)
		
		#embedded = [batch size, emb dim, sent len]

		contextual_embedded = torch.cat((self.unigram(embedded), self.trigram(embedded), self.fivegram(embedded)), 1)
		
		#contextual_embedded = [batch size, emb dim, sent len]
		
		# sent_emb = self.pool(torch.transpose(contextual_embedded, 1, 2)).squeeze(2)
		
		sent_emb = torch.mean(contextual_embedded, 2)
		
		#sent_emb = [batch size, emb dim]
		
		output = self.out(sent_emb)
		
		#output = [batch size, out dim]
		
		return output, sent_emb, torch.transpose(embedded, 1, 2)
	

class MutualInformation(nn.Module):
	def __init__(self, sent_pad_idx):        
		super().__init__()

		self.sent_pad_idx = sent_pad_idx

		self.m = nn.LogSigmoid()

	def create_mask(self, sent):
		mask = (sent != self.sent_pad_idx)#.permute(1, 0)
		return mask

	def forward(self, sent, sent_len, sent_emb, token_emb):

		#sent = [batch size, sent len]
		#sent_len = [batch size]
		#sent_emb = [batch size, emb dim]
		#token_emb = [batch size, sent len, emb dim]

		sent_emb = torch.unsqueeze(sent_emb, 2)

		#sent_emb = [batch size, emb dim, 1]

		inner = self.m(torch.bmm(token_emb, sent_emb).squeeze(2))

		#inner = [batch size, sent len]

		mask = self.create_mask(sent)

		#mask = [batch size, sent len]

		inner = inner.masked_fill(mask == 0, 1e-10)

		inner = torch.sum(inner, 1) / torch.sum(mask, 1)

		#inner = [batch size]

		return inner


