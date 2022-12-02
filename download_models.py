import transformers
import torch


def main():
    torch.save(transformers.BertModel.from_pretrained(
        "bert-base-uncased"), "data/bert.pt")

    torch.save(transformers.BigBirdModel.from_pretrained(
        "google/bigbird-roberta-base"), "data/bigbird.pt")

    torch.save(transformers.GPT2Model.from_pretrained("gpt2"), "data/gpt2.pt")


if __name__ == "__main__":
    main()
