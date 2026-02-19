# ASFNet



## Model Weights (Baidu Pan)

- **Link**: https://pan.baidu.com/s/1lqGCH99o7aK-QybVLnRcvQ
- **Code**: `mfiy`

## Install

```bash
conda create -n asfnet python=3.8
conda activate asfnet
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Dataset

Download links:
- **ISPRS Vaihingen**: https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx
- **ISPRS Potsdam**: https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx
- **LoveDA**: https://codalab.lisn.upsaclay.fr/competitions/421


## Train

```bash
python train_supervision.py -c config/vaihingen/ours.py
python train_supervision.py -c config/potsdam/ours.py
python train_supervision.py -c config/loveda/ours.py
```

## Test

```bash
python vaihingen_test.py -c config/vaihingen/ours.py
python potsdam_test.py -c config/potsdam/ours.py
python loveda_test.py -c config/loveda/ours.py
```



