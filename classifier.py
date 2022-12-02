from transformers import BertTokenizer, BertModel
import torch

labels = {'business': 0,
          'entertainment': 1,
          'sport': 2,
          'tech': 3,
          'politics': 4
          }

key_list = list(labels.keys())
val_list = list(labels.values())


class Data(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.texts = [tokenizer(text,
                                padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def __len__(self):
        return len(self.texts)

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        return batch_texts


class BertClassifier(torch.nn.Module):
    def __init__(self, bert, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = bert
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(768, 5)
        self.relu = torch.nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


def get_labels(values):
    keys = []
    for value in values:
        position = val_list.index(value)
        keys.append(key_list[position])
    return keys


def infer(bert_model, tokenizer, data):
    infer_output = []
    batch_len = min(len(data), 16)
    test = Data(data, tokenizer)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_len)

    model = BertClassifier(bert_model)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    with torch.no_grad():
        for test_input in test_dataloader:
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            if batch_len > 1:
                er_output += output.argmax(dim=1).squeeze().tolist()
            else:
                infer_output += output.argmax(dim=1)

    return infer_output
