安装命令
```Powershell
conda create -n mmocr2023 python=3.8 -y
conda activate mmocr2023

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet==3.1.0
```