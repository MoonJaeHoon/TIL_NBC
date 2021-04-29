import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

def plot_conf_mat_origin(train_y_true,train_y_pred):
  cm = confusion_matrix(train_y_true,train_y_pred)
  df_cm = pd.DataFrame(cm, columns=np.unique(train_y_true), index = np.unique(train_y_true))
  df_cm.index.name = 'Actual'
  df_cm.columns.name = 'Predicted'

  # neptune save confusion matrix 
  tr_conf_mat_figure = plt.figure(figsize=(40,36))
  sns.set(font_scale=1.4) #for label size

  sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 18}, fmt="d")# font size
  # plt.title(f"CONFUSION MATRIX : ")
  plt.ylabel('True Label')
  plt.xlabel('Predicted label')
  return tr_conf_mat_figure

def plot_conf_mat_normalized(train_y_true,train_y_pred):
  cm = confusion_matrix(train_y_true,train_y_pred)
  column_sum_cm = np.sum(cm,axis=1)
  df_cm = pd.DataFrame(cm/column_sum_cm[:,None], columns=np.unique(train_y_true), index = np.unique(train_y_true))
  df_cm.index.name = 'Actual'
  df_cm.columns.name = 'Predicted'

  # neptune save confusion matrix 
  tr_conf_mat_figure_normalized = plt.figure(figsize=(40,36))
  sns.set(font_scale=1.4) #for label size
  sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 18})# font size
  # plt.title(f"CONFUSION MATRIX : ")
  plt.ylabel('True Label')
  plt.xlabel('Predicted label')
  return tr_conf_mat_figure_normalized
