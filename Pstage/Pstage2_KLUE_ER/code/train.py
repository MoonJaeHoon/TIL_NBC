import pickle as pickle
import os
import pandas as pd
import numpy as np
import random
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, AutoModelForSequenceClassification, AutoConfig
from transformers import AdamW, get_constant_schedule, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, EarlyStoppingCallback
from load_data import *
import argparse
from importlib import import_module
from pathlib import Path
import glob
import re
import neptune
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from CustomizedCosineAnnealingWarmRestarts import CustomizedCosineAnnealingWarmRestarts
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from tqdm import tqdm
from collections import defaultdict
from k_th_plot_from_logs import k_th_plot_from_logs
import warnings
warnings.filterwarnings(action='ignore')  # 'default'

project_qualified_name = 'jh951229/Pstage-2-EntityRelationExtraction'
API_Token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMmYwY2E4Ny0yYTg5LTRiZmQtODNjZC1mMzRmN2Q5ODFkNDkifQ=='

neptune.init(project_qualified_name=project_qualified_name,api_token=API_Token)

# seed 고정 
def seed_everything(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # if use multi-GPU
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  os.environ["PYTHONHASHSEED"] = str(seed)
  np.random.seed(seed)
  random.seed(seed)

# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)

  return {
      'accuracy': acc,
  }

def increment_output_dir(output_path, exist_ok=False):
  
  path = Path(f'./results/{output_path}')
  # print(path)
  if (path.exists() and exist_ok) or (not path.exists()):
    return str(path)
  else:
    dirs = glob.glob(f"{path}*")
    matches = [re.search(rf"%s(\d+)" %path.stem, dr) for dr in dirs]
    i = [int(m.groups()[0]) for m in matches if m]
    n = max(i) + 1 if i else 1
    return f"{path}{n}"


def lower_dir_search(dirname):  # 모델저장된 체크포인트 폴더 경로들을 찾아주는 함수
  filenames = os.listdir(dirname)
  lower_dir_list = []
  for filename in filenames:
      full_filename = os.path.join(dirname, filename)
      lower_dir_list.append(full_filename)
  return lower_dir_list

def train(args):  # + inference 과정까지 추가하였습니다.
  assert sum([args.use_kfold,args.use_simple_fold,args.no_valid])==1
  assert (args.concat_exp_p==0 or args.concat_log_p==0)
  # assert args.eval_steps == args.logging_steps
  if args.use_kfold==True:
    assert (args.num_fold_k>=2)

  seed_everything(args.seed)
  USE_KFOLD = args.use_kfold
  # load model and tokenizer
  model_type_getattr = args.model_type  # ELECTRA # BERT
  model_name_from_pretrained = args.pretrained_model # "monologg/koelectra-small-discriminator", "monologg/koelectra-small-discriminator"
  tokenizer = AutoTokenizer.from_pretrained(model_name_from_pretrained)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # load dataset
  # load_data_module = getattr(import_module("load_data"), f'load_data')
  # dataset = load_tr_val_data("../input/data/train/train.tsv",args)
  dataset = load_tr_val_data("../input/data/train/final_train_ner.tsv",args)

  # setting model hyperparameter
  # config_module = getattr(import_module("transformers"), f'{args.model_type}Config')
  # model_config = config_module.from_pretrained(model_name_from_pretrained)
  model_module = getattr(import_module("transformers"), f'{model_type_getattr}ForSequenceClassification')
  model = model_module.from_pretrained(model_name_from_pretrained, num_labels=42)
  model.parameters
  model.to(device)

  # model_saved_dir = increment_output_dir(args.model_output_dir)
  model_saved_dir = increment_output_dir(model_name_from_pretrained.replace('/','_')) # f'./results/{output_path}'

  neptune.append_tag(f"{model_saved_dir.split('/')[-1]}")
  neptune.append_tag(f"{args.name}")


  with open('../input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)

  # Simple Train Valid Split
  # Not KFOLD # => StratifiedShuffleSplit

  #################################################################################################
  #################################################################################################
  elif args.use_kfold==True: # KFOLD
    if not os.path.isdir('./kfold_results'):  # 모델들을 저장할 상위폴더
      os.makedirs('./kfold_results')
    kfold = StratifiedKFold(n_splits=args.num_fold_k, random_state=args.seed, shuffle=True)
    label = dataset['label']

    # 이미 해당 모델로 kfold가 수행되고 모델 저장된 적이 있는지 확인
    model_name_from_pretrained_used_for_save = model_name_from_pretrained.replace('/','_')
    check_upper_dir = f'./kfold_results/{model_name_from_pretrained_used_for_save}'
    if not os.path.isdir(check_upper_dir+'0'):  # 존재하지 않는다면 그대로 사용
      upper_dir=check_upper_dir+'0'
    else: # 존재한다면 존재하는 것들 중 숫자 찾아서 최댓값 +1 을 사용
      all_directories = glob.glob(f'./kfold_results/*')
      max_num = max(int(re.search(rf"{model_name_from_pretrained_used_for_save}[0-9]+",ad).group().replace(model_name_from_pretrained_used_for_save,'')) for ad in all_directories if re.search(rf"{model_name_from_pretrained_used_for_save}[0-9]+",ad))
      upper_dir = check_upper_dir+str(max_num+1)

    neptune.log_text('Model_Name_Number', f"{upper_dir.split('/')[-1]}")

    kfold_train_acc_score = []
    kfold_val_acc_score = []

    k=0
    for train_idx, val_idx in kfold.split(dataset, label):
      # model_module = getattr(import_module("transformers"), f'{model_type_getattr}ForSequenceClassification')
      # model = model_module.from_pretrained(model_name_from_pretrained, num_labels=42)
      config_module = getattr(import_module("transformers"), f'{model_type_getattr}Config')
      model_config = config_module.from_pretrained(model_name_from_pretrained)
      # model_config = ElectraConfig.from_pretrained(model_name_from_pretrained)
      model_config.num_labels = 42
      model_config.hidden_dropout_prob = args.hidden_dropout_prob
      model_module = getattr(import_module("transformers"), f'{model_type_getattr}ForSequenceClassification')
      model = model_module.from_pretrained(model_name_from_pretrained, config=model_config)

      model.parameters
      model.to(device)
      print('='*50)
      print('=' * 15 + f'{k}-th Fold Cross Validation Started ({k+1}/{args.num_fold_k})' + '=' * 15)

      train_dataset = dataset.iloc[train_idx]
      val_dataset = dataset.iloc[val_idx]

      # 새로운 외부데이터 추가해서 학습해보기
      if args.concat_external_data==True:
        train_dataset = concat_external_data(train_dataset,label_type,args)


      train_label = train_dataset['label'].values
      val_label = val_dataset['label'].values
      
      # tokenizing dataset
      tokenized_train = tokenized_dataset(train_dataset, tokenizer, args)
      tokenized_val = tokenized_dataset(val_dataset, tokenizer, args)

      # make dataset for pytorch.
      RE_train_dataset = RE_Dataset(tokenized_train, train_label)
      RE_val_dataset = RE_Dataset(tokenized_val, val_label)
      print('='*50)
      print('Train & Valid Loaded Successfully!!')
      print(f'len(RE_train_dataset) : {len(RE_train_dataset)}, len(RE_val_dataset) : {len(RE_val_dataset)}')
      
      model_saved_dir = upper_dir+f'/{k}fold' # f'./kfold_results/{model_name_from_pretrained_used_for_save}'+f'/{k}fold'
      neptune.log_text(f'{k}-th model_saved_dir',model_saved_dir)
      neptune.log_text(f'Num_Data : {k}-th len(RE_train_dataset)',str(len(RE_train_dataset)))
      neptune.log_text(f'Num_Data : {k}-th len(RE_val_dataset)',str(len(RE_val_dataset)))

      # 사용한 option 외에도 다양한 option들이 있습니다.
      # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
      # https://huggingface.co/transformers/main_classes/trainer.html?highlight=trainingarguments#trainingarguments

      total_num_epochs = (len(RE_train_dataset)//args.batch_size+1)*args.epochs
      if args.use_warmup_ratio:
        warmup_steps = total_num_epochs*args.warmup_ratio
      else:
        warmup_steps = 0
      wandb_run_name = model_saved_dir.replace('./kfold_results/',f'Total {args.num_fold_k}fold :')
      training_args = TrainingArguments(
          report_to = 'wandb', # 'all'
          run_name = f"{args.name+wandb_run_name.replace('/','_')}",
          output_dir=model_saved_dir,          # output directory
          # overwrite_output_dir=False, # 모델을 저장할 때 덮어쓰기 할 것인지
          save_total_limit=args.save_total_limit,              # number of total save model.
          save_steps=args.model_save_steps,                 # model saving step.
          num_train_epochs=args.epochs,              # total number of training epochs
          learning_rate=args.lr,               # learning_rate
          per_device_train_batch_size=args.batch_size,  # batch size per device during training
          per_device_eval_batch_size=args.val_batch_size,   # batch size for evaluation
          warmup_steps=warmup_steps,                # number of warmup steps for learning rate scheduler
          weight_decay=args.weight_decay,               # strength of weight decay
          logging_dir='./logs',            # directory for storing logs
          logging_steps=args.logging_steps,              # log saving step.
          evaluation_strategy='steps', # evaluation strategy to adopt during training
          eval_steps = args.eval_steps,            # evaluation step.
          # max_grad_norm=1,
          label_smoothing_factor = args.label_smoothing_factor,
          load_best_model_at_end = args.load_best_model_at_end,  # default => False
          # greater_is_better = True,
          metric_for_best_model = args.metric_for_best_model, # metric_for_best_model: Optional[str] = None
          # fp16 = True,  # Whether to use 16-bit (mixed) precision training instead of 32-bit training.
          # dataloader_num_workers = 2,
        )

      # EarlyStopping
      # 여기선 global epochs 하이퍼파라미터를 기준으로 하지 않고, Total_Step을 본다.
      # 만약 patience를 1로 설정하면 eval_step * 1만큼을 기준으로 판단한다. (eval_step=25로 설정했다면 25만큼 patience)
      early_stopping = EarlyStoppingCallback(
                                            early_stopping_patience = args.early_stopping_patience, 
                                            early_stopping_threshold = 1e-4)

      ## Optimizer
      if args.optimizer_name == "Adam":
        optimizer = Adam(model.parameters(), lr=args.min_lr)
      elif args.optimizer_name == "AdamW":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
      elif args.optimizer_name == "SGD":
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
      
      # https://arxiv.org/pdf/1608.03983.pdf
      ## Scheduler
      T_0 = int(np.ceil(total_num_epochs*args.first_cycle_ratio))
      if args.scheduler_name == "Custom":
        scheduler = CustomizedCosineAnnealingWarmRestarts(optimizer,
                                                          T_0=T_0,
                                                          T_mult=2,
                                                          eta_max=args.lr,
                                                          T_up=int(T_0*args.first_warmup_ratio), 
                                                          gamma=args.scheduler_gamma,
                                                          last_epoch=-1)
      elif args.scheduler_name == "Original":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2, eta_min=args.min_lr)

      # https://huggingface.co/tkransformers/main_classes/trainer.html?highlight=trainer#id1
      trainer = Trainer(
        model=model,                         # Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_val_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,         # define metrics function
        optimizers=  (optimizer,scheduler), # optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]=(None, None))
        callbacks=  [early_stopping], # callbacks: Optional[List[TrainerCallback]]=None
        # model_init= 
      )
      # train model
      trainer.train()

      print(f'Neptune Saving {k}-th Model Logs Plot')
      # Get Log from Model
      # Neptune Save Plot (train_eval_loss, learning_rate, eval_accuracy)
      train_eval_loss_plot, learning_rate_plot, eval_accuracy_plot = k_th_plot_from_logs(trainer.state.log_history)
      neptune.log_image(f'Logs : {k}-th train_eval_loss_plot',train_eval_loss_plot)
      neptune.log_image(f'Logs : {k}-th learning_rate_plot',learning_rate_plot)
      neptune.log_image(f'Logs : {k}-th eval_accuracy_plot',eval_accuracy_plot)

      print(f'{k}-th train finished!!')


      state_log_history = trainer.state.log_history
      eval_log_dict = [log_dict for log_dict in state_log_history if 'eval_loss' in log_dict.keys() ]
      k_th_val_logs_dict = defaultdict(list)
      for dict_per_step in eval_log_dict:
        for key,value in dict_per_step.items():
          k_th_val_logs_dict[key].append(value)

      best_val_acc_score = max(k_th_val_logs_dict['eval_accuracy'])

      # neptune.log_metric(f'{k}-th train_acc_score',best_train_acc_score)
      neptune.log_metric(f'{k}-th val_acc_score',best_val_acc_score)

      kfold_val_acc_score.append(best_val_acc_score)
      k=int(k)
      k+=1
      
    # neptune.log_text(f"{args.num_fold_k}-fold train best acc list", f"{kfold_train_acc_score}")
    neptune.log_text(f"{args.num_fold_k}-fold val best acc list", f"{kfold_val_acc_score}")
    # neptune.log_metric(f"Result ACC : {args.num_fold_k}-fold train Total Average acc", np.mean(kfold_train_acc_score))
    neptune.log_metric(f"Result ACC : {args.num_fold_k}-fold val Total Average acc", np.mean(kfold_val_acc_score))



def main(args):
  train(args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--name' , type=str , default = 'nlp_model')
  parser.add_argument('--seed' , type=int , default = 77)
  # parser.add_argument('--model_type', type=str, default='Bert')
  # parser.add_argument('--model_type', type=str, default='Albert')
  # parser.add_argument('--model_type', type=str, default='DistilBert')
  # parser.add_argument('--model_type', type=str, default='GPT2')
  # parser.add_argument('--model_type', type=str, default='OpenAIGPT')
  parser.add_argument('--model_type', type=str, default='XLMRoberta')
  # parser.add_argument('--model_type', type=str, default='Electra')
  # parser.add_argument('--pretrained_model', type=str, default='bert-base-multilingual-cased')
  # parser.add_argument('--pretrained_model', type=str, default="monologg/kobert")
  # parser.add_argument('--pretrained_model', type=str, default="monologg/kobert-lm")
  # parser.add_argument('--pretrained_model', type=str, default="kykim/bert-kor-base")
  # parser.add_argument('--pretrained_model', type=str, default="monologg/distilkobert")
  # parser.add_argument('--pretrained_model', type=str, default="monologg/koelectra-base-finetuned-naver-ner")
  # parser.add_argument('--pretrained_model', type=str, default="monologg/koelectra-small-discriminator")
  # parser.add_argument('--pretrained_model', type=str, default="monologg/koelectra-small-v2-discriminator")
  # parser.add_argument('--pretrained_model', type=str, default="monologg/koelectra-small-v3-discriminator")
  # parser.add_argument('--pretrained_model', type=str, default="monologg/koelectra-base-discriminator")
  # parser.add_argument('--pretrained_model', type=str, default="monologg/koelectra-base-v2-discriminator")
  # parser.add_argument('--pretrained_model', type=str, default="monologg/koelectra-base-v3-discriminator")
  # parser.add_argument('--pretrained_model', type=str, default="monologg/koelectra-base-v3-naver-ner")
  # parser.add_argument('--pretrained_model', type=str, default="monologg/koelectra-mecab-wp-small-discriminator")
  # parser.add_argument('--pretrained_model', type=str, default="kykim/funnel-kor-base")
  parser.add_argument('--pretrained_model', type=str, default="xlm-roberta-large")
  parser.add_argument('--label_smoothing_factor', type=float, default=0.1)
  parser.add_argument('--val_ratio', type=float, default=0)
  # parser.add_argument('--lr', type=float, default=5e-5)
  parser.add_argument('--lr', type=float, default=1e-5)
  parser.add_argument('--min_lr', type=float, default=1e-7)
  parser.add_argument('--num_fold_k', type=int, default=5)
  parser.add_argument('--epochs', type=int, default=20)
  # parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--val_batch_size', type=int, default=64)
  parser.add_argument('--weight_decay', type=float, default=0.01)
  # parser.add_argument('--warmup_steps', type=int, default=0)               # number of warmup steps for learning rate scheduler
  parser.add_argument('--first_warmup_ratio', type=float, default=0.2)               # number of warmup steps for learning rate scheduler
  parser.add_argument('--warmup_ratio', type=float, default=0.0)               # number of warmup steps for learning rate scheduler
  # parser.add_argument('--scheduler_gamma', type=float, default=0.9)               # number of warmup steps for learning rate scheduler (0.9, 0.)
  parser.add_argument('--scheduler_gamma', type=float, default=1.0)               # number of warmup steps for learning rate scheduler (0.9, 0.)
  # parser.add_argument('--first_cycle_ratio', type=float, default=0.035) # first cycle ratio for learning rate scheduler # (0.035 -> 약 5번, 0.07 -> 약 4번, 0.15 -> 약 3번)
  parser.add_argument('--first_cycle_ratio', type=float, default=0.1) # first cycle ratio for learning rate scheduler # (0.035 -> 약 5번, 0.07 -> 약 4번, 0.15 -> 약 3번)
  parser.add_argument('--scheduler_name', type=str, default='Custom')               # number of warmup steps for learning rate scheduler
  parser.add_argument('--optimizer_name', type=str, default='Adam')               # number of warmup steps for learning rate scheduler
  # parser.add_argument('--model_output_dir', type=str, default='./results/model')
  parser.add_argument('--model_save_steps', type=int, default=100)
  parser.add_argument('--save_total_limit', type=int, default=1)
  parser.add_argument('--eval_steps', type=int, default=25)
  parser.add_argument('--early_stopping_patience', type=int, default=50)
  parser.add_argument('--logging_steps', type=int, default=25)
  parser.add_argument('--logging_dir', type=str, default='./logs')            # directory for storing logs
  parser.add_argument('--metric_for_best_model', type=str, default='eval_accuracy') # default = None
  parser.add_argument('--max_len', type=int, default=128) 
  parser.add_argument('--apply_add_entity', type=bool, default=False)
  parser.add_argument('--load_best_model_at_end', type=bool, default=True)
  parser.add_argument('--use_warmup_ratio', type=bool, default=True)               # number of warmup steps for learning rate scheduler

  parser.add_argument('--use_kfold', type=bool, default=True)
  parser.add_argument('--use_simple_fold', type=bool, default=False)
  parser.add_argument('--no_valid', type=bool, default=False)
  parser.add_argument('--concat_external_data', type=bool, default=True)
  parser.add_argument('--concat_len_filter', type=bool, default=False)  # 외부 데이터 가져올 때, 기존 sentence의 len보다 길면 필터링
  parser.add_argument('--concat_exp_p', type=float, default=0.0)
  parser.add_argument('--concat_log_p', type=float, default=2.0)  # concat_external 함수에서의 로그 밑, 1에서 2 사이의 값으로 설정하기, 값이 작게 할 수록 샘플 추가를 많이 하겠다는 뜻
  parser.add_argument('--use_papago', type=bool, default=False)  # 파파고 번역을 통한 augmentation
  parser.add_argument('--hidden_dropout_prob', type=float, default=0.1) # 크게 가져가는게 우리 Task에 좋을듯

  args = parser.parse_args()
  neptune.create_experiment(name=f"{args.pretrained_model.replace('/','_')}", params=vars(args))
  main(args)

# python train.py --name kodelectra_base_v3_176_Test_concat_expp_2
# python train.py --concat_exp_p 2.0 --scheduler_gamma 0.95 --name kodelectra_base_v3_concat_expp_2__gamma_0dot95_dropout_default
# python train.py --name XLM_ROBERTA0_concat_logp2_patience_50_lr1e-5_min_lr5e-8
# python train.py --name XLM_ROBERTA_Final_Processing
# tensorboard --logdir=./logs
# !df -h
# Wandb 사용하기 : 현재 제공하는 서버 기준에서는 원래 transformers 를 version 4.2.0로 사용중이다. -> wandb 사용을 위해서는 transformers 4.5.0이 필요.
# pip uninstall transformers
# pip install transformers==4.5.0

# [사용 용량 확인]
# cd /opt/
# du -h --max-depth=1 | sort -hr

# [휴지통 비우기]
# cd ~
# rm -r ./.local/share/Trash/files