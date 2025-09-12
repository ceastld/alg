# 卷积神经网络 (CNN)

## 概述

卷积神经网络（Convolutional Neural Networks, CNN）是一类深度神经网络，特别适用于处理图像等网格状数据。它们使用卷积操作自动学习特征的空间层次结构，使其成为计算机视觉任务的标准架构。

## 核心概念

### 1. 卷积操作

卷积操作在输入上应用滤波器（核）来产生特征图。

**数学定义：**
$$(f * g)(t) = \int f(\tau)g(t - \tau)d\tau$$

对于离散的二维卷积：
$$(I * K)(i,j) = \sum_m\sum_n I(m,n)K(i-m, j-n)$$

### 2. 关键组件

#### 卷积层
- **滤波器/核**：检测特定特征的小矩阵
- **特征图**：卷积操作的输出
- **步长**：滑动滤波器时的步长
- **填充**：在输入周围添加零来控制输出大小

#### 池化层
- **最大池化**：取每个区域的最大值
- **平均池化**：取每个区域的平均值
- **目的**：减少空间维度和计算负载

#### 激活函数
- **ReLU**：$f(x) = \max(0, x)$ - CNN 中最常用
- **Leaky ReLU**：$f(x) = \max(0.01x, x)$ - 解决 ReLU 死亡问题
- **Swish**：$f(x) = x \cdot \text{sigmoid}(x)$ - ReLU 的平滑替代

## 架构演进

### LeNet-5 (1998)
- 第一个成功用于数字识别的 CNN
- 2 个卷积层 + 3 个全连接层
- 引入了基本的 CNN 结构

### AlexNet (2012)
- ImageNet 竞赛的突破性成果
- 5 个卷积层 + 3 个全连接层
- ReLU 激活、dropout、数据增强
- GPU 实现

### VGG (2014)
- 非常深的网络（16-19 层）
- 全程使用小的 3×3 滤波器
- 简单、统一的架构
- 强大的特征提取能力

### ResNet (2015)
- 残差连接解决梯度消失问题
- 跳跃连接：$H(x) = F(x) + x$
- 能够训练非常深的网络（100+ 层）
- 批量归一化

### DenseNet (2017)
- 所有层之间的密集连接
- 特征重用和梯度流
- 参数效率

## 现代架构

### EfficientNet (2019)
- 深度、宽度和分辨率的复合缩放
- 高效使用参数和计算资源
- 用更少参数达到最先进的精度

### Vision Transformer (ViT) (2020)
- 将 Transformer 架构应用于图像
- 将图像块视为序列标记
- 在大数据集上与 CNN 竞争

## 数学基础

### 前向传播
$$y = \sigma(W * x + b)$$

其中：
- $W$：权重矩阵（滤波器）
- $x$：输入
- $b$：偏置
- $\sigma$：激活函数
- $*$：卷积操作

### CNN 中的反向传播
- 梯度通过卷积操作流动
- 权重更新使用局部梯度
- 池化层将梯度传递给最大位置

## 实现细节

### 填充策略
- **Valid**：无填充，输出大小 = 输入大小 - 滤波器大小 + 1
- **Same**：填充以保持输入大小
- **Full**：最大填充以获得完全重叠

### 步长效果
- **步长 = 1**：密集卷积，最大重叠
- **步长 > 1**：稀疏采样，减少输出大小

### 批量归一化
- 归一化每层的输入
- 减少内部协变量偏移
- 支持更高的学习率
- 作为正则化

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 批量归一化
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # 第一个卷积块
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 第二个卷积块
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 第三个卷积块
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # 展平
        x = x.view(-1, 128 * 4 * 4)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差块
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
```

## 应用领域

### 计算机视觉
- **图像分类**：CIFAR-10、ImageNet
- **目标检测**：R-CNN、YOLO、SSD
- **语义分割**：FCN、U-Net、DeepLab
- **人脸识别**：FaceNet、ArcFace

### 医学影像
- **X 射线分析**：肺炎检测
- **MRI/CT 扫描**：肿瘤检测
- **视网膜成像**：糖尿病视网膜病变

### 其他领域
- **自然语言处理**：文本分类
- **时间序列**：股票预测、天气预报
- **音频处理**：语音识别、音乐分类

## 训练技巧

### 数据增强
- **几何变换**：旋转、平移、缩放、翻转
- **颜色调整**：亮度、对比度、饱和度调整
- **噪声**：高斯噪声、dropout
- **Mixup/CutMix**：高级增强技术

### 正则化
- **Dropout**：随机置零神经元
- **批量归一化**：归一化层输入
- **权重衰减**：L2 正则化
- **早停**：防止过拟合

### 优化
- **Adam**：自适应学习率
- **带动量的 SGD**：传统优化
- **学习率调度**：随时间降低学习率
- **梯度裁剪**：防止梯度爆炸

## 性能考虑

### 计算效率
- **参数共享**：卷积减少参数
- **稀疏连接**：仅局部连接
- **平移不变性**：在任何地方检测相同特征

### 内存使用
- **特征图**：存储中间激活
- **梯度存储**：反向传播所需
- **模型大小**：参数数量

## 最新进展

### 注意力机制
- **Squeeze-and-Excitation**：通道注意力
- **CBAM**：卷积块注意力模块
- **自注意力**：非局部操作

### 高效架构
- **MobileNet**：深度可分离卷积
- **ShuffleNet**：通道混洗以提高效率
- **EfficientNet**：复合缩放

### 神经架构搜索 (NAS)
- **AutoML**：自动化架构设计
- **DARTS**：可微分架构搜索
- **EfficientNet**：NAS 发现的架构

## 参考文献

1. LeCun, Y., et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 1998.
2. Krizhevsky, A., et al. "Imagenet classification with deep convolutional neural networks." NIPS 2012.
3. Simonyan, K., & Zisserman, A. "Very deep convolutional networks for large-scale image recognition." ICLR 2015.
4. He, K., et al. "Deep residual learning for image recognition." CVPR 2016.
5. Huang, G., et al. "Densely connected convolutional networks." CVPR 2017.
