from peft import PeftConfig, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, RobertaForMaskedLM, default_data_collator, get_linear_schedule_with_warmup
import torch
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

''' Model Config '''

model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

''' Data '''

text_column = 'description'
label_column = 'target'

# set description max length 
max_length = 128

def make_dataset(path):
    ''' 
    This function creates transformers Datasets for a given csv file that consists of the columns 'descriptions' and 'targets' 
    '''
    df = pd.read_csv(path)
    print(f"Number of descriptions: {len(df)}")
    print(f"Number of target words: {df['target'].nunique()}")

    # Create Transformers dataset
    data = {"target": [word for word in df['target']], "description":[descr for descr in df['description']]}
    dataset = Dataset.from_dict(data)
    data_split = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    train = data_split['train']
    dev_test = data_split['test'].train_test_split(test_size=0.5, shuffle=True, seed=42)
    dev = dev_test['train']
    test = dev_test['test']
    del data_split['test']
    data_split['eval'] = dev
    data_split['test'] = test 

    return train, dev, test, data_split

def preprocess_masked(examples):
    '''
    This function preprocesses a given dataset for masked language modelling. This approach has been only followed test wise, thus it is not considered in the main prompt tuning file
    '''
    batch_size = len(examples[text_column])
    inputs = [f"Description : {x} Word :" for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        sample_input_ids.remove(tokenizer.eos_token_id)
        label_ids = labels["input_ids"][i]
        label_ids.remove(tokenizer.bos_token_id)
        label_ids_masked = [tokenizer.mask_token_id if id not in tokenizer.all_special_ids else id for id in label_ids]
        model_inputs["input_ids"][i] = sample_input_ids + label_ids_masked
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_ids 
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def preprocess_seq2seq(examples):
    '''
    This function preprocesses a given dataset. It embeds each target-description pair into the format 'Description: ... Word: ...'
    '''
    inputs = examples[text_column]
    targets = examples[label_column]
    target_max_length = max([len(tokenizer(target)["input_ids"]) for target in targets])
    prefix = "Description: "
    suffix = " Word: "
    inputs = [prefix + input_text + suffix for input_text in inputs]

    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=target_max_length, padding="max_length", truncation=True, return_tensors="pt")
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


''' Training '''

def training(train_dataloader, eval_dataloader, model, num_epochs, lr=0.1, device="cuda:6"):
    '''
    This function executes one training procedure. 
    '''
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)
    
    model = model.to(device)

    train_preds = {}
    for epoch in range(num_epochs):
        print(f"-----Epoch {epoch+1} has started-----")
        model.train()
        total_loss = 0
        epoch_preds = []
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            epoch_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        train_preds[f"{epoch+1}"] = epoch_preds

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
    
    model.save_pretrained("/local/js/BERT_Friends")

    return train_preds, eval_preds
    
''' Evaluation '''

def get_peft_model_inference(peft_model_id):
    '''
    This function returns model and tokenizer for a peft model at inference.
    '''
    config = PeftConfig.from_pretrained(peft_model_id)

    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    if next(model.parameters()).is_cuda:
        print("Model is on GPU")
    else:
        print("Model is on CPU")

    return model, tokenizer

def acc(targets, preds):
    '''
    This function returns the top-k accuracy for a given set of targets and predictions.
    '''
    correct_top_k = 0
    for target, top_k_pred in zip(targets, preds):
       if target in top_k_pred:
           correct_top_k += 1
    
    print(f"Top-k correct predictions: {correct_top_k}")
    
    return correct_top_k/len(targets)
    
def eval_seq2seq(data, test_dataloader, model, peft_model_id=None, num_beams=1, num_seqs=1, device="cuda:0", zero_shot=False, file_spec=None):
    ''' 
    This function evaluates a seq2seq PEFT model for a given test dataset. It returns targets and predictions.
    '''
    if not zero_shot and peft_model_id:
        model = PeftModel.from_pretrained(model, peft_model_id)
    
    model.to(device)

    if next(model.parameters()).is_cuda:
        print("Model is on GPU")
    else:
        print("Model is on CPU")

    if num_seqs > num_beams:
        num_seqs = num_beams

    predictions = []

    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            beam_outputs = model.generate(
                input_ids=batch['input_ids'], max_new_tokens=10, num_beams=num_beams, num_return_sequences=num_seqs, early_stopping=True
            )
            
            beam_preds = []
            for beam_output in beam_outputs:
                prediction = tokenizer.decode(beam_output.detach().cpu().numpy(), skip_special_tokens=True).strip().lower()
                beam_preds.append(prediction)
            predictions.append(beam_preds)

    if file_spec:
        df_eval_data = pd.DataFrame({'description': data['description'], 'target': data['target'], 'prediction': predictions})
        df_eval_data.to_csv(file_spec)

    return data['target'], predictions

def maskedLM_eval(data, model_name, file_spec, zero_shot=False, k=1, test_dataloader=None):
    '''
    This function evaluates a PEFT model for masked LM. This has only been used test wise. Thus, it is not considered in the main prompt tuning file.
    '''
    if zero_shot:
        model = RobertaForMaskedLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        device = "cuda:6"
        model.to(device)

        predictions = []

        model.eval()

        for target, description in zip(data['target'], data['description']):
        #for inputs in data:

            seq_masked = f"Description: {description} Word: <mask>"
            inputs = tokenizer(seq_masked, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits

            # retrieve index of <mask>
            mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

            predicted_token_id = logits[0, mask_token_index, :]
            top_k_tokens = torch.topk(predicted_token_id, k, dim=1).indices[0].tolist()
            top_k_decoded = []
            for token in top_k_tokens:
                pred_token = tokenizer.decode([token], skip_special_tokens=True).strip().lower()
                top_k_decoded.append(pred_token)
            predictions.append(top_k_decoded)  
        
        df_eval_data = pd.DataFrame({'description': data['description'], 'target': data['target'], 'prediction': predictions})
        df_eval_data.to_csv(f"/local/js/BERT_Friends/eval_data_{file_spec}.csv")

        return data['target'], predictions
    
    else:

        peft_model_id = "/local/js/BERT_Friends"
        config = PeftConfig.from_pretrained(peft_model_id)
        model = RobertaForMaskedLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, peft_model_id)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

        device = "cuda:0"
        model.to(device)

        model.eval()

        preds_test = []
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Perform masked language modeling inference to predict the masked elements
            with torch.no_grad():
                outputs = model(**batch)

            # Get the top-k predicted token IDs for the masked positions
            k = 5 
            mask_token_index = (batch['input_ids'] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            pred_ids = outputs.logits[0, mask_token_index, :]
            topk_predicted_masked_token_ids = torch.topk(pred_ids, k=k, dim=1).indices

            masked_seqs = [[] for _ in range(k)]
            for i in range(len(topk_predicted_masked_token_ids)):
                for j in range(len(topk_predicted_masked_token_ids[i])):
                    masked_seqs[j].append(topk_predicted_masked_token_ids[i][j])

            # Convert the top-k predicted token IDs to tokens
            topk_predicted_masked_tokens = tokenizer.batch_decode(masked_seqs, skip_special_tokens=True)
            topk_predicted_masked_tokens = [token.strip().lower() for token in topk_predicted_masked_tokens]

            preds_test.append(topk_predicted_masked_tokens)
        
        df_eval_data = pd.DataFrame({'description': data['description'], 'target': data['target'], 'prediction': preds_test})
        df_eval_data.to_csv(f"/local/js/BERT_Friends/eval_data_{file_spec}.csv")
        
        return data['target'], preds_test
