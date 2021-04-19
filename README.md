# self-supervised-sentence-embedding-model-pytorch
a model based on transformers

## Usage
To use the models here, you may want to check out this [Tutorial](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/6%20-%20Transformers%20for%20Sentiment%20Analysis.ipynb). Majority of codes are adapted there.

We so far include two forward models and one loss function model here. The forward models work in the same manner, and the difference is their architecture. They both take in a text tensor and output a binary prediction, a sentence embedding vector and vectors of tokens in the sentence. Of course, these are all in batches.

Since we are to calculate the mutual information between a sentence vector and each of its token vectors, a better practice is to set ```include_lengths``` as ```True``` like this. (**Note** All functions or variables not defined here can be found in the tutorial above)

```python
from torchtext.legacy import data

TEXT = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx,
                  include_lengths = True)

TEXTNEG = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx,
                  include_lengths = True)
```
Then you can load your dataset as usual
```python
fields = {'your_text_field': ('text', TEXT), 'your_negative_text_field': ('text_neg', TEXTNEG)}
```

and then import the models like this
```python
from models import BERTGRU, MutualInformation, BERTCNN
```
You may choose to use either forward model like this
```python
from transformers import BertTokenizer, BertModel
bert = BertModel.from_pretrained('bert-base-uncased')

OUTPUT_DIM = 1

model = BERTCNN(bert, OUTPUT_DIM)
```
or this
```python
HIDDEN_DIM = 384 # = 768 / 2
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = BERTGRUSentiment(bert,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT)
```
Then your mutual information function can be loaded like this
```python
mutual = MutualInformation(pad_token_idx).to(device)
```

The training and evaluating function need to be modified a bit from the tutorial
```python
def train(model, iterator, optimizer, criterion, mode = 'mi'):
    
	epoch_loss = 0
	epoch_acc = 0

	model.train()

	for batch in iterator:

		optimizer.zero_grad()

		text, sent_len = batch.text

		predictions, hidden, embedded = model(text)

		text_neg, sent_len_neg = batch.text_neg

		predictions_neg, hidden_neg, embedded_neg = model(text_neg)

		predictions = predictions.squeeze(1)

		predictions_neg = predictions_neg.squeeze(1)

		mutual_loss = torch.mean(
			mutual(text_neg, sent_len_neg, hidden, embedded_neg) - 
			mutual(text, sent_len, hidden, embedded))

		if mode == 'binary':

			loss = criterion(predictions, batch.label)

		elif mode == 'mi':

			loss = mutual_loss

		elif mode == 'double':

			loss = criterion(predictions, batch.label) + criterion(predictions_neg, batch.label_neg)

		else:

			loss = mutual_loss * 3e-2 + criterion(predictions, batch.label) + criterion(predictions_neg, batch.label_neg)

		acc = binary_accuracy(predictions, batch.label)

		loss.backward()

		optimizer.step()

		epoch_loss += loss.item()
		epoch_acc += acc.item()
		print(loss.item())
	return epoch_loss / len(iterator), epoch_acc / len(iterator)
```

```python
def evaluate(model, iterator, criterion, mode = 'mi'):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text, sent_len = batch.text

            predictions, hidden, embedded = model(text)

            text_neg, sent_len_neg = batch.text_neg

            predictions_neg, _, embedded_neg = model(text_neg)

            predictions = predictions.squeeze(1)

            predictions_neg = predictions_neg.squeeze(1)

            mutual_loss = torch.mean(
                mutual(text_neg, sent_len_neg, hidden, embedded_neg) - 
                mutual(text, sent_len, hidden, embedded))

            if mode == 'binary':

                loss = criterion(predictions, batch.label)

            elif mode == 'mi':

                loss = mutual_loss

            elif mode == 'double':

                loss = criterion(predictions, batch.label) + criterion(predictions_neg, batch.label_neg)

            else:

                loss = mutual_loss * 3e-2 + criterion(predictions, batch.label) + criterion(predictions_neg, batch.label_neg)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
```

Everything should be fine once you patch these scripts to the original [Tutorial](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/6%20-%20Transformers%20for%20Sentiment%20Analysis.ipynb). Then, you will get your sentence embedding trained by maximizing the inner product with word vectors in this sentence and minimizing the inner product with word vectors **not** in this sentence.

As you may have noticed, you could load the same pretrained architecture for a binary classification later.

## After words
Selecting correct pair of sentence and negative sentence pairs could be a tricky issue, you may figure this out when you implement your own task. We use randomly selected sentences in our case.
