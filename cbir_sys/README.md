## 1 Introduction
cbir_sys is a system for image retrieval, including demo and method evaluation.

## 2 System demo


## 3 Evaluation
### 3.1 roxford5k
| method | mAP | mp@k[1 5 10] |
| ------ | ------ | ------ |
| vgg16_ImageNet | E: 36.44, M: 27.1, H: 8.03 | E: [57.35 51.91 47.52], M: [57.14 51.43 44.71], H: [22.86 17.93 14.94] |
| R–GeM[^1]  | E: 84.81, M: 64.67, H: 38.47 | E: [97.06 92.06 86.49], M: [97.14 90.67 84.67], H: [81.43 63.   53.  ] |

### 3.2 rparis6k
| method | mAP | mp@k[1 5 10] |
| ------ | ------ | ------ |
| vgg16_ImageNet | 64.25, M: 49.73, H: 21.99 | E: [94.29 89.57 88.  ], M: [97.14 91.71 90.86], H: [71.43 56.   49.71] |
| R–GeM[^1] | E: 92.12, M: 77.2, H: 56.32 | E: [100.    97.14  96.14], M: [100.    98.86  98.14], H: [94.29 90.29 89.14] |


### Reference
[^1]: F. Radenović, G. Tolias, and O. Chum. Fine-tuning CNN
image retrieval with no human annotation. In arXiv, 2017.