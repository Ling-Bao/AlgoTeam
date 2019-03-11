# kf
[参考资料](https://mp.weixin.qq.com/s?__biz=MzI1NjkxOTMyNQ==&mid=2247486367&idx=1&sn=809b181e9cb54d3f268e065fe31b8071&chksm=ea1e19eddd6990fba657edbcc7545aa1119b7043c046af830f19046f64ae0ce4e162a5d3129c&scene=0&xtrack=1&pass_ticket=PSlzXdpUytcU33%2BWx4gDvi88GIydoLGIDxBPfIlALrlsV7ZRl%2FvmntH6nspPadlp#rd)

## 1 依赖项
- Ceres
- Eigen

## 2 简化卡尔曼滤波器
### 2.1 初始化
&emsp;&emsp;初始化：实现各个变量的初始化，对于不同的运动模型，其状态向量存在差异。以在2D空间运行的小车为例，需要x/y方向上的距离和速度来表示小车的状态。
```
void Initialization(Eigen::VectorXd x_in);
```

### 2.2 预测
&emsp;&emsp;预测：$x'=Fx+u, P'=FPF^T+Q$
```
void Predict();
```

### 2.3 观测
