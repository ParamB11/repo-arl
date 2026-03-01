## Prerequisites
Clone the repository:
```
git clone https://github.com/ParamB11/repo-arl.git --recursive
cd repo-arl
mkdir outputs
```

Install [CARL](https://github.com/automl/CARL) and [policy_transfer](https://github.com/VincentYu68/policy_transfer/tree/master).

## How to use
The training procedure can be divided into training the Universal Policy and training the context estimator.
To train the UP use the code [train_up_carl.py](https://github.com/ParamB11/repo-arl/blob/main/src/train_up_carl.py) and to train the context estimator use the code train_osi_ver3.py
An example of training and evaluation can be found in scripts. Run it using the command:
`bash scripts\pendulumscript.sh`