# A simple Kalman Filter
[Reference Resource](https://mp.weixin.qq.com/s?__biz=MzI1NjkxOTMyNQ==&mid=2247486367&idx=1&sn=809b181e9cb54d3f268e065fe31b8071&chksm=ea1e19eddd6990fba657edbcc7545aa1119b7043c046af830f19046f64ae0ce4e162a5d3129c&scene=0&xtrack=1&pass_ticket=PSlzXdpUytcU33%2BWx4gDvi88GIydoLGIDxBPfIlALrlsV7ZRl%2FvmntH6nspPadlp#rd)

## 1 Dependencies
- Ceres
- Eigen

## 2 An example of simple kelman filter
### 2.1 Initialization
&emsp;&emsp;Initialize：For different motion models, the state vectors are different. Take the car running in 2D space as an example, the distance and speed in the x/y direction are needed to represent the state of the car.
```
void Initialization(Eigen::VectorXd x_in);
```

### 2.2 Prediction
&emsp;&emsp;Predict：$x'=Fx+u, P'=FPF^T+Q$
```
void Predict();
```

### 2.3 Observation
