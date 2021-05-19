## torch 모델 저장한 모델로 다시 학습하기





> 출처 : https://tutorials.pytorch.kr/beginner/saving_loading_models.html



1. model.state_dict만 저장하던 이전과는 달리 다음과 같이 저장해주어야할 변수들이 더 많다.

```python
def save_checkpoint(epoch,model,optimizer,loss):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, file_name)	# 이와 같이 저장할 때에는 file_name의 확장자명을 보통 .tar로 해준다.
```



2. 그리고 다음과 같이 model을 load하고 train() 혹은 eval()을 수행해주면 된다.

```python
# load
# https://tutorials.pytorch.kr/beginner/saving_loading_models.html

checkpoint = torch.load(file_name)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']


############################
model.train()
model.eval()
```

