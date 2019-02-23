# AlgoTeam
for algorithm sharing

## 1 cbir_sys
- Introduction
```
This is system of content-based image retrieval
```

- Dependencies
```
# python3.6 and minianacod3

cd AlgoTeam
pip install -r requirements.txt

# install faiss
# https://github.com/facebookresearch/faiss/blob/master/INSTALL.md
conda install faiss-cpu -c pytorch
```

- Starting
```
# Download vgg16 pretrain model from BaiduYunPan
# 链接: https://pan.baidu.com/s/1xMfA5yXA04N6JbTxzlWL3g
# 提取码: cv86

cd AlgoTeam
cp vgg16.h5 cbir_sys/support/

cd cbir_sys
python index_image_tool.py 
```
