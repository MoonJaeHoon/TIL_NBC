import pickle as pickle
import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import random
# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
  def __init__(self, tokenized_dataset, labels):
    self.tokenized_dataset = tokenized_dataset
    self.labels = labels

  def __getitem__(self, idx):
    # item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
    item = {key: val[idx].clone().detach() for key, val in self.tokenized_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type, args):
  label = []
  for i in dataset.iloc[:,8]:
    if i == 'blind':
      label.append(100)
    else:
      label.append(label_type[i])
  out_dataset = pd.DataFrame({'sentence':dataset.iloc[:,1],'entity_01':dataset.iloc[:,2],'entity_02':dataset.iloc[:,5],'label':label,})
  # if args.apply_add_entity == True:
  #   out_dataset['sentence'] = out_dataset.apply(lambda x: x['entity_01']+"[SEP]"+x['entity_02']+"[SEP]"+x['sentence'], axis=1)
  return out_dataset

def preprocessing_oversampling_simple_fold_dataset(dataset, label_type, args):
  # random.seed(args.seed)
  # if args.concat_external_data == True:
  #   process_external_data = pd.read_csv("../input/data/process_external_data/all_csv.tsv",sep='\t',header=None)
  #   min_freq = process_external_data[8].value_counts()[-1]
  #   chosen_index = []
  #   for process_unique_label in set(process_external_data[8]):
  #     chosen_index.extend(random.sample(process_external_data.loc[process_external_data[8]==process_unique_label].index.tolist(),min_freq))
  #   process_external_data = process_external_data.iloc[chosen_index]
  #   dataset = pd.concat([dataset,process_external_data],axis=0)
  #   dataset.index = range(dataset.shape[0])
  #   print('='*50)
  #   print("external_data concat완료")
  #   print(f"dataset.shape : {dataset.shape}")
  #   print('='*50)
  # print(f"dataset : {dataset}")
  # print(f"label_type : {label_type}")

  label = []
  for i in dataset.iloc[:,8]:
    # print(f"i : {i}")
    if i == 'blind':
      label.append(100)
    else:
      label.append(label_type[i])
  out_dataset = pd.DataFrame({'sentence':dataset.iloc[:,1],'entity_01':dataset.iloc[:,2],'entity_02':dataset.iloc[:,5],'label':label,})
  
  # # 단순오버샘플링
  # label = out_dataset.label
  # if args.use_simple_fold:
  #   threshold = int(1/args.val_ratio)
  # elif args.use_kfold:
  #   threshold = int(args.num_fold_k)*2

  # need_oversample_label = label.value_counts()[label.value_counts()<threshold].index
  # need_oversample_label = np.ceil(threshold/label.value_counts()[need_oversample_label])
  # need_oversample_label = dict(need_oversample_label)

  # for label,oversample in need_oversample_label.items():
  #   data_from_label = out_dataset.loc[out_dataset.label==label]
  #   for _ in range(int(oversample)):
  #     out_dataset = pd.concat([out_dataset,data_from_label],axis=0)
  #   out_dataset.index = range(out_dataset.shape[0])
  
  return out_dataset

# tsv 파일을 불러옵니다.
# def load_data(dataset_dir,args):
#   # load label_type, classes
#   with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
#     label_type = pickle.load(f)
#   # load dataset
#   dataset = pd.read_csv(dataset_dir, delimiter='\t')
#   # preprecessing dataset
#   dataset = preprocessing_dataset(dataset, label_type)
  
#   return dataset

def concat_external_data(train_dataset,label_type,args):

  random.seed(args.seed)
  # 규진님이 주신 데이터 추가해보기
  # 현재 train데이터의 라벨 분포 비율을 고려하여 데이터를 추가한다.
  # process_external_data = pd.read_csv("../input/data/process_external_data/all_csv.tsv",sep='\t',header=None)
  process_external_data = pd.read_csv("../input/data/external_data/final_external_ner.tsv",sep='\t')
  process_external_data = preprocessing_dataset(process_external_data, label_type, args)
  # process_vc = process_external_data.loc[:,'label'].value_counts()
  # train_vc = train_dataset.loc[:,'label'].value_counts()

  # min_freq = process_vc.iloc[-1]
  # # print(min_freq)
  # vc_train_intersection_process = train_vc[process_vc.index].sort_values(ascending=False)
  # min_freq_name = process_vc.index[process_vc.argmin()]
  # num_chosen_dict = {}
  # min_freq = process_vc[min_freq_name]
  # num_chosen_dict[min_freq_name] = min_freq

  # threshold = vc_train_intersection_process[min_freq_name]
  # log_p = args.concat_log_p
  # exp_p = args.concat_exp_p
  # for freq_name in [pv_id for pv_id in process_vc.index if pv_id!=min_freq_name]:
  #   # min_freq/current_freq, 데이터 존재비율 반영하여 라벨별 데이터 수 다르게 추가
  #   count_freq_per_min = threshold/vc_train_intersection_process[freq_name]
  #   ## 로그함수 이용 : 성능 좋았지만, p를 2보다 더 크게 조정할 수만 있다는 한계점
  #   if log_p != 0:
  #     transformed_sample_prop = np.log(count_freq_per_min+(log_p-1))/np.log(log_p)  # 0보다 큰 값이 나와야 함.
  #   elif exp_p !=0:
  #     transformed_sample_prop = exp_p**count_freq_per_min/exp_p  # 0보다 큰 값이 나와야 함.
  #   num_chosen_dict[freq_name] = min(int(np.ceil(transformed_sample_prop * min_freq)), process_vc[freq_name])
  # print(f'num_chosen_dict : {num_chosen_dict}')
  # chosen_index = []
  # for process_unique_label in set(process_external_data['label']):
  #   chosen_index.extend(random.sample(process_external_data.loc[process_external_data['label']==process_unique_label].index.tolist(),num_chosen_dict[process_unique_label]))
  # process_external_data = process_external_data.iloc[chosen_index]
  train_dataset = pd.concat([train_dataset,process_external_data],axis=0)
  train_dataset.index = range(train_dataset.shape[0])

  return train_dataset


def load_test(dataset_dir,args):
  # load label_type, classes
  with open('../input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)
  # load dataset
  dataset = pd.read_csv(dataset_dir, delimiter='\t')
  # preprecessing dataset
  dataset = preprocessing_dataset(dataset, label_type, args)
  
  return dataset

def load_tr_val_data(dataset_dir,args):
  # load label_type, classes
  with open('../input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)
  # load dataset
  dataset = pd.read_csv(dataset_dir, sep='\t')
  # preprecessing dataset
  if args.use_kfold:
    dataset = preprocessing_oversampling_simple_fold_dataset(dataset, label_type, args)
  elif args.use_simple_fold:
    dataset = preprocessing_oversampling_simple_fold_dataset(dataset, label_type, args)
  else:
    dataset = preprocessing_dataset(dataset, label_type, args)
    
  return dataset

# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenized_dataset(dataset, tokenizer,args):
  concat_entity = []
  if 'Roberta' in args.model_type:
    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
      temp = ''
      temp = '<s>' + e01 + '</s>' + '<s>' + e02 + '</s>'
      concat_entity.append(temp)
  else:
    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
      temp = ''
      temp = e01 + '[SEP]' + e02
      concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      # max_length=100,
      max_length=args.max_len,

      add_special_tokens=True,
      )
  return tokenized_sentences
