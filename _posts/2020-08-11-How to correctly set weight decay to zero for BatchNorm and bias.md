---
layout: post
title:  "How to correctly set weight decay to zero for BatchNorm and bias in Pytorch"
---

Recently I participated in the Kaggle's [ALASKA2 Image Steganalysis contest](https://www.kaggle.com/c/alaska2-image-steganalysis). While trying to gather new ideas, I've stumbled upon an [interesting notebook](https://www.kaggle.com/shonenkov/train-inference-gpu-baseline) by a top Notebook Grandmaster. However, I observed something didn't look right:

```
param_optimizer = list(self.model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.002},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
] 

self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
```

As you can see, he passes the original parameters (`self.model.parameters()`) to the optimizer instead of the modified ones (`optimizer_grouped_parameters`)

In fact, I did two experiments:

For each experiment I used these two lines of code to print the optimizer's parameter groups:
```
for param_group in optimizer.param_groups:
    print("Current weight decay: ", param_group['weight_decay'])
```

1. `optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)`

    ```
    Total number of parameters trained this epoch:  4012672
    Current weight decay:  0.01
    ```

2. `optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)`

    ```
    Total number of parameters trained this epoch:  4012672
    Current weight decay:  0.002
    Current weight decay:  0.0
    ```

As you can see, in both of the prints, the same number of parameters are trained (so no parameters were lost by passing `optimizer_grouped_parameters`), but in the first one, the weight decay is 0.01 (the default for AdamW), while in the second print there are two separate weight decays.