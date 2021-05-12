## ImportError: cannot import name 'SAVE_STATE_WARNING' from 'torch.optim.lr_scheduler' (/opt/conda/lib/python3.7/site-packages/torch/optim/lr_scheduler.py)



이와 같은 에러가 발생했을 때,



/opt/conda/lib/python3.7/site-packages/transformers/trainer_pt_utils.py 의 37번째 line을 수정해야 함



> 원래의 코드

```python
if version.parse(torch.__version__) <= version.parse("1.4.1"):
    SAVE_STATE_WARNING = ""
else:
    from torch.optim.lr_scheduler import SAVE_STATE_WARNING
    
logger = logging.get_logger(__name__)

```



> 수정 후 코드

```python
# if version.parse(torch.__version__) <= version.parse("1.4.1"):
#     SAVE_STATE_WARNING = ""
# else:
#     from torch.optim.lr_scheduler import SAVE_STATE_WARNING

try:
    from torch.optim.lr_scheduler import SAVE_STATE_WARNING
except ImportError:
    SAVE_STATE_WARNING = ""
    
logger = logging.get_logger(__name__)

```

