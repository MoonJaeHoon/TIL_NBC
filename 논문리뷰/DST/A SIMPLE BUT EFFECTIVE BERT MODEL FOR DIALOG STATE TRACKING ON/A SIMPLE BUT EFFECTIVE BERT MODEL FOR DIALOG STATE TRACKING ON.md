# A SIMPLE BUT EFFECTIVE BERT MODEL FOR DIALOG STATE TRACKING ON RESOURCE-LIMITED SYSTEMS

> https://arxiv.org/pdf/1910.12995v3.pdf



## ABSTRACT

In this work, we propose a simple but effective DST model based on BERT

In addition to its simplicity, our approach also has a number of other advantages:

​	(a) the number of parameters does not grow with the ontology size

​	(b) the model can operate in situations where the domain ontology may change dynamically



> To make the model small and fast enough for resource-restricted systems, we apply the knowledge distillation method to compress our model



## 1. INTRODUCTION

many neural network based approaches have been proposed for the task of DST [2, 3, 4, 5, 6, 7, 8, 9]

These methods achieve highly competitive performance on standard DST datasets such as DSTC-2 [10] or WoZ 2.0 [11]

However, most of these methods still have some limitations

1.  many approaches require training a separate model for each slot type in the domain ontology [2, 4, 7].
   - Therefore, the number of parameters is proportional to the number of slot types, making the scalability of these approaches a significant issue
2. some methods only operate on a fixed domain ontology [3, 4]
   - The slot types and possible values need to be defined in advance and must not change dynamically
3. state-of-the-art neural architectures for DST are typically heavily-engineered and conceptually complex [4, 5, 6, 7]



In this paper, we show that by finetuning a pretrained BERT model, we can build a conceptually simple but effective model for DST.

Given a dialog context and a candidate slot-value pair, the model outputs a score indicating the relevance of the candidate

Because the model shares parameters across all slot types, the number of parameters does not grow with the ontology size

Furthermore, because each candidate slot-value pair is simply treated as a sequence of words, the model can be directly applied to new types of slot-value pairs not seen during training

 We do not need to retrain the model every time the domain ontology changes





<img src="A%20SIMPLE%20BUT%20EFFECTIVE%20BERT%20MODEL%20FOR%20DIALOG%20STATE%20TRACKING%20ON.assets/image-20210504143705170.png" alt="image-20210504143705170" style="zoom:150%;" />