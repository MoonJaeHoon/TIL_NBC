from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse

# import argparse
from importlib import import_module

def inference_proba(model, tokenized_sent, device, args):
  dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
  model.eval()
  output_pred = []
  result=np.zeros(42).reshape(-1,42)

  for i, data in enumerate(dataloader):
    with torch.no_grad():
      if 'Roberta' in args.model_type:
        outputs = model(
            input_ids=data['input_ids'],
            attention_mask=data['attention_mask'],
            # token_type_ids=data['token_type_ids']
            )
      else:
        outputs = model(
            input_ids=data['input_ids'],
            attention_mask=data['attention_mask'],
            token_type_ids=data['token_type_ids']
            )
      # outputs = model(
      #     input_ids=data['input_ids'].to(device),
      #     attention_mask=data['attention_mask'].to(device),
      #     token_type_ids=data['token_type_ids'].to(device)
      #     )
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    # result = np.argmax(logits, axis=-1)
    result=np.vstack([result,logits])

  return result[1:,:]

def inference(model, tokenized_sent, device, args):
  dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
  model.eval()
  output_pred = []
  
  for i, data in enumerate(dataloader):
    with torch.no_grad():
      if 'Roberta' in args.model_type:
        outputs = model(
            input_ids=data['input_ids'],
            attention_mask=data['attention_mask'],
            # token_type_ids=data['token_type_ids']
            )
      else:
        outputs = model(
            input_ids=data['input_ids'],
            attention_mask=data['attention_mask'],
            token_type_ids=data['token_type_ids']
            )
      # outputs = model(
      #     input_ids=data['input_ids'].to(device),
      #     attention_mask=data['attention_mask'].to(device),
      #     token_type_ids=data['token_type_ids'].to(device)
      #     )
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)
    output_pred.append(result)
  from itertools import chain
  output_pred = list(chain(*output_pred))

  return np.array(output_pred).flatten()

def load_test_dataset(dataset_dir, tokenizer,args):
  test_dataset = load_test(dataset_dir,args)

  test_label = test_dataset['label'].values
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer,args)
  return tokenized_test, test_label

def main(args):
  """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  # TOK_NAME = "bert-base-multilingual-cased" 
  TOK_NAME = args.pretrained_model
  tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)

  # load my model
  model_module = getattr(import_module("transformers"), args.model_type + "ForSequenceClassification")
  model = model_module.from_pretrained(args.model_dir)
  model.parameters
  model.to(device)

  # load test datset
  test_dataset_dir = "../input/data/test/test.tsv"
  test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  test_dataset = RE_Dataset(test_dataset ,test_label)

  # predict answer
  pred_answer = inference(model, test_dataset, device)
  # make csv file with predicted answer
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.

  output = pd.DataFrame(pred_answer, columns=['pred'])
  #   output.to_csv('./prediction/submission.csv', index=False)
  
  if os.path.exists(args.out_path):
    output.to_csv(args.out_path, index=False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # model dir
  parser.add_argument('--model_dir', type=str, default="./results/bert-base-multilingual-cased0/checkpoint-2000")
  parser.add_argument('--out_path', type=str, default="../prediction/submission.csv")
  parser.add_argument('--model_type', type=str, default="Bert")
  parser.add_argument('--pretrained_model', type=str, default="bert-base-multilingual-cased")
  args = parser.parse_args()
  print(args)
  main(args)