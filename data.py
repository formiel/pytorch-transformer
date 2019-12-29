import torchtext
from torchtext.data.utils import get_tokenizer

TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)

def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data

def get_batch(source, i, bptt=35):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def get_loaders(name, batch_size):
    if name == 'WikiText2':
        train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
        TEXT.build_vocab(train_txt)
    else:
        raise NotImplementedError

    train_data = batchify(train_txt, batch_size)
    val_data = batchify(val_txt, batch_size)
    test_data = batchify(test_txt, batch_size)

    dataloaders = {'train': train_data, 'val': val_data, 'test': test_data}
    # the size of vocabulary
    ntokens = len(TEXT.vocab.stoi)
    info = {'ntokens': ntokens}
    return dataloaders, info