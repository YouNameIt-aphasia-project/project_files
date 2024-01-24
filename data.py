from transformers import AutoTokenizer, default_data_collator
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader

''' Model Config '''

model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

''' Data '''

def_col = 'definitions'
word_col = 'word'

# set defintion max length 
max_length = 128

def make_dataset(path, top1_def=False, randomk=None):
    '''
    This function creates a transformers dataset for a given file.
    '''
    data = pd.read_json(path)

    # consider only the first definition starting from the top as this is the most prominent one
    if top1_def:
        data = data.drop_duplicates(subset=[word_col])
    
    # consider k samples out of the total data
    elif randomk:
        data = data.sample(n=randomk, random_state=42)
    
    elif top1_def and randomk:
        data = data.drop_duplicates(subset=[word_col])
        data = data.sample(n=randomk, random_state=42)

    data = {"word": [word for word in data[word_col]], "definitions":[definition for definition in data[def_col]]}
    return Dataset.from_dict(data)

def preprocess_seq2seq(examples):
    '''
    This function preprocesses a given dataset. It embeds each word-definition pair into the format 'Definition: ... Word: ...'
    '''
    inputs = examples[def_col]
    words = examples[word_col]
    word_max_length = max([len(tokenizer(word)["input_ids"]) for word in words])
    prefix = "Definition: "
    suffix = " Word: "
    inputs = [prefix + input_text + suffix for input_text in inputs]

    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(words, max_length=word_max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    
    return model_inputs

def make_dataloader(dataset_preproc, batch_size):
    '''
    This function creates PyTorch dataloader for a given preprocessed dataset and batch size
    '''
    train_dataset = dataset_preproc["train"]
    eval_dataset = dataset_preproc["eval"]
    test_dataset = dataset_preproc["test"]

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=1, pin_memory=True)

    return train_dataloader, eval_dataloader, test_dataloader