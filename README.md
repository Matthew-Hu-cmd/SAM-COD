# SAM-COD  
基于prompt-engineering、one-shot transfer learing将Meta AI的工作*segment-anything*迁移到伪装目标检测的任务中
```
 requirements：python = 3.8 torch = 1.8.0 
 CUDA = 11.1 torchvision = 0.9.0 python-opencv
 segment-anything source code and pre-trained weights
 ```

 并不公平的对比：案例可见```test-cod.ipynb```,是否有必要将第一次的mask再次和gt对比得出遗漏的区域并在用互补的bbox对齐进行预测，mask补全  