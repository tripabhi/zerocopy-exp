import torch
import ray
import utils
import transformers
import classifier
import pandas as pd


class ModelHandler():

    def __init__(self):
        print("Initializing ModelHandler")
        self.models = ["bert", "bigbird", "gpt2"]
        self.tokenizer_dict = {}
        self.db = {}
        self.store_models_in_store()
        self.download_tokenizers()
        print("Finished initializing ModelHandler")

    def store_models_in_store(self):
        for m in self.models:
            fname = f'./data/{m}.pt'
            model = torch.load(fname)
            self.db[m] = ray.put(utils.extract_weights(model))

    def download_tokenizers(self):
        self.tokenizer_dict["bert"] = transformers.BertTokenizerFast.from_pretrained(
            "bert-base-uncased")
        self.tokenizer_dict["bigbird"] = transformers.BigBirdTokenizer.from_pretrained(
            "google/bigbird-roberta-base")
        self.tokenizer_dict["gpt2"] = transformers.GPT2Tokenizer.from_pretrained(
            "gpt2")

    def get_input_dataframe(self, input):
        infer_data = []
        infer_data.append(input)
        return pd.DataFrame(infer_data, columns=['text'])

    def handle(self, request):
        model = request["model"]
        input = request["input"]
        model_ref = self.db[model]
        skeleton, weights = ray.get(model_ref)
        utils.install_weights(skeleton, weights)

        tokenizer = self.tokenizer_dict[model]
        input_df = self.get_input_dataframe(input)
        output = classifier.infer(skeleton, tokenizer, input_df)
        labels = classifier.get_labels(output)

        return labels
