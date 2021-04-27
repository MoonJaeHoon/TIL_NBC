from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def k_th_plot_from_logs(state_log_history):
  # k_th_logs_dict['step']  # x축 # k_th_logs_dict['epoch']
  want_plot_params = ['loss', 'learning_rate','eval_loss', 'eval_accuracy'] # y축이 될 변수들

  train_log_dict = [log_dict for log_dict in state_log_history if 'learning_rate' in log_dict.keys() ]
  eval_log_dict = [log_dict for log_dict in state_log_history if 'eval_loss' in log_dict.keys() ]

  k_th_tr_logs_dict = defaultdict(list)
  for dict_per_step in train_log_dict:
    for k,v in dict_per_step.items():
      k_th_tr_logs_dict[k].append(v)

  k_th_val_logs_dict = defaultdict(list)
  for dict_per_step in eval_log_dict:
    for k,v in dict_per_step.items():
      k_th_val_logs_dict[k].append(v)

  tr_x_ticks = k_th_tr_logs_dict['step'][:len(k_th_tr_logs_dict['loss'])]
  val_x_ticks = k_th_val_logs_dict['step'][:len(k_th_val_logs_dict['eval_loss'])]
#   print('='*50)
#   print(k_th_tr_logs_dict['loss'])
#   print('='*50)

  # Loss Plot
  train_eval_loss_plot = plt.figure(figsize=(20,12))
  sns.set(font_scale=1.4)
  plt.plot(tr_x_ticks, k_th_tr_logs_dict['loss'],
          label="Train Loss")
  plt.plot(val_x_ticks, k_th_val_logs_dict['eval_loss'],
          label="Val Loss")
  plt.xlabel('Num_Steps')
  plt.ylabel('Loss')
  plt.legend() # 꼭 호출해 주어야만 legend가 달립니다
  # plt.show()

  # learning rate Plot
  learning_rate_plot = plt.figure(figsize=(20,12))
  sns.set(font_scale=1.4)
  plt.plot(tr_x_ticks, k_th_tr_logs_dict['learning_rate'],
          label="learning rate")
  plt.xlabel('Num_Steps')
  plt.ylabel('learning rate')
  plt.legend() # 꼭 호출해 주어야만 legend가 달립니다
  # plt.show()

  # eval accuracy Plot
  eval_accuracy_plot = plt.figure(figsize=(20,12))
  sns.set(font_scale=1.4)
  plt.plot(val_x_ticks, k_th_val_logs_dict['eval_accuracy'],
          label="eval accuracy")
  plt.xlabel('Num_Steps')
  plt.ylabel('eval accuracy')
  plt.legend() # 꼭 호출해 주어야만 legend가 달립니다
  # plt.show()

  return train_eval_loss_plot, learning_rate_plot, eval_accuracy_plot