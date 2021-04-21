import torch

class Predictor():

	def __init__(self):
		self.model = None
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def count_parameters(self):
		return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

	def freeze_parameters(self):
		print(f'The model has {self.count_parameters():,} trainable parameters')
		for name, param in self.model.named_parameters():                
			param.requires_grad = False
		print(f'The model has {self.count_parameters():,} trainable parameters')
		return None

	def stoi(self, sentence):
		return None

	def output(self, input):
		return None		

	def __call__(self, sentence):
		self.model.eval()		
		indexed = self.stoi(sentence)		
		tensor = torch.LongTensor(indexed).to(self.device)
		tensor = tensor.unsqueeze(0)

		return self.output(tensor)
