import torch
import nltk
import pickle
import pandas as pd
nltk.download('wordnet')
nltk.download('punkt')
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def compute_sentence_embeddings(definitions_wordNet, out_file, device):
  '''
  This function computes sentence embeddings for a given database.
  '''
  model.to(device)
  embeddings = model.encode(definitions_wordNet)

  #Store sentences & embeddings on disc
  with open(out_file, "wb") as fOut:
      pickle.dump({'sentences': definitions_wordNet, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)


def get_sentence_embeddings(path, device):
  ''' 
  This function loads stored sentences and embeddings
  '''
  with open(path, "rb") as fIn:
      stored_data = pickle.load(fIn)
      stored_sentences = stored_data['sentences']
      stored_embeddings = torch.tensor(stored_data['embeddings'], device=device)
  
  return stored_sentences, stored_embeddings


def generate_target_transformers(stored_embeddings, user_input, targets, model=model, k=10):
  ''' 
  This function returns the top-k most similar target given a user description.
  '''
  #Compute embedding for user input
  emb_user = model.encode(user_input, convert_to_tensor=True)

  #Compute cosine-similarities for user input sentence with each other sentence; use stored_embeddings from pickle file
  cosine_scores = util.cos_sim(emb_user, stored_embeddings)

  _ , indices = torch.sort(cosine_scores[0], descending=True)
  topKindices = indices[:k]

  topKtargets = [targets[idx] for idx in topKindices]

  return topKtargets

def preds(descriptions, targets, sent_embeddings, targets_total, k=10):
  ''' 
  This function returns the predictions for a given file containing description-target pairs.
  '''
  #df = pd.read_csv(file)
  golds_predictions = []
  gold_in_pred = 0
  for description, target in zip(descriptions, targets):
    target = target.lower()
    k_preds = generate_target_transformers(sent_embeddings, description, targets_total, k=k)
    k_preds = [pred.split('.')[0] for pred in k_preds]
    golds_predictions.append((target, k_preds))
    if target in k_preds:
      gold_in_pred += 1

  acc_k = gold_in_pred / len(targets)

  return golds_predictions, acc_k

def top_k_acc(descriptions, targets, targets_total, sent_embeddings):
  '''
  This function returns the top-1 to top-10 accuracy of a given set of descriptions.
  '''
  topk_acc_dict = dict()
  eval = dict()
  for k in range(1,11):
    golds_preds, acc_k = preds(descriptions, targets, sent_embeddings, targets_total, k=k)
    eval[k] = golds_preds
    topk_acc_dict[k] = acc_k

  return topk_acc_dict
