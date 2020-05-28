# Meta-Learning with Warped Gradient Descent Reproducability Study
This repository contains the modified code from the authors of Meta-Learning with Warped Gradient Descent (Flennerhag, S., et al.) along with scripts used to reproduce the results outlined in the paper, which was publish at ICLR 2020. Our report on the reproducibility of the experiments is also included.

**Original Paper**: Flennerhag, S., Rusu, A.A., Pascanu, R., Yin, H. and Hadsell, R., 2019. Meta-Learning with Warped Gradient Descent. arXiv preprint arXiv:1909.00025, Available: https://openreview.net/pdf?id=rkeiQlBFPB

**Original Code**: https://github.com/flennerhag/warpgrad

## Running Notes
To reproduce our reduced meta training parmeter results the number of pretraining tasks and a name for logging must be specified (in this example, 10 pretraining tasks are used and the logging name is "run_name"):
```
./run_reduced_meta_params 10 run_name
```
This is also the case for the reduced task parameters:
```
./run_reduced_task_params 10 run_name
```

## Authors
* **Jack Dymond** [jd5u19@soton.ac.uk]()
* **Hsuan-Yang Wang** [hw1g11@soton.ac.uk]()
* **Tom Kelly** [tgk2g14@soton.ac.uk]()
