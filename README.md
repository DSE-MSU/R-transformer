# R-Transformer
Pytorch implementation of [R-Transformer](https://arxiv.org/abs/1907.05572).  Some parts of the code are adapted from the implementation of [TCN](https://github.com/locuslab/TCN) and [Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html). 


For more details about R-Transformer, Please read our [paper](https://arxiv.org/abs/1907.05572).  If you find this work useful and use it on your research, please cite our paper.

```
@article{wang2019rtransf,
  title={R-Transformer: Recurrent Neural Network Enhanced Transformer},
  author={Wang, Zhiwei and Ma, Yao and Liu, Zitao and Tang, Jiliang},
  journal={arXiv preprint arXiv:1907.05572},
  year={2019}
}
```

## Usage
Our repository is arranged as follows:
```
[Task Name] /
    data/ # contains the datasets
    experiment.py #run experiment 
    model.py # comtains the model
    utils.py # utility functions including dataset downloading and preprocessing
models /
    RTransformer.py # RTransformer model    
```
The dataset for the "polyphonic music modeling" experiment is already included in audio/data/. For other experiments that are based on much larger datasets, the data needs to be downloaded (from [torchvision.datasets](https://pytorch.org/docs/stable/torchvision/datasets.html) or [observations](https://github.com/edwardlib/observations)) and then put into the "data" folder which should be created firstly.

When data is ready, the code can directly run with PyTorch  1.0.0.
## Final Words
We will keep this repo updated and hope it is useful to your research. 
