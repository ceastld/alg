# Transformer 架构

## 概述

Transformer 架构在《Attention Is All You Need》（Vaswani et al., 2017）中首次提出，通过用自注意力机制替代循环和卷积层，彻底改变了自然语言处理领域。它已成为许多最先进模型的基础，包括 BERT、GPT 和 T5。

## 核心组件

### 1. 自注意力机制

自注意力机制允许模型在处理每个位置时关注输入序列的不同部分。

**数学公式：**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q$：查询矩阵
- $K$：键矩阵
- $V$：值矩阵
- $d_k$：键向量的维度

### 2. 多头注意力

多头注意力并行运行多个注意力机制，允许模型关注不同的表示子空间。

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中每个头为：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 3. 位置编码

由于 Transformer 没有固有的序列顺序，需要向输入嵌入添加位置编码：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

### 4. 前馈网络

每一层都包含一个位置相关的前馈网络：

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

## 架构细节

### 编码器-解码器结构

**编码器：**
- 6 个相同的层
- 每层包含：多头注意力 + 前馈网络
- 残差连接和层归一化

**解码器：**
- 6 个相同的层
- 每层包含：掩码多头注意力 + 编码器-解码器注意力 + 前馈网络
- 掩码注意力防止训练时看到未来的标记

### 层归一化

在每个子层之前（Pre-LN）或之后（Post-LN）应用：
$$\text{LayerNorm}(x + \text{Sublayer}(x))$$

## 优势

1. **并行化**：与 RNN 不同，所有位置可以同时处理
2. **长距离依赖**：远距离位置之间的直接连接
3. **可解释性**：注意力权重提供模型决策的洞察
4. **可扩展性**：性能随数据和参数增加而提升

## 应用领域

- **语言建模**：GPT 系列
- **机器翻译**：原始 Transformer 论文
- **文本分类**：BERT
- **图像处理**：Vision Transformer (ViT)
- **多模态**：CLIP、DALL-E

## 实现考虑

### 计算复杂度
- 自注意力：$O(n^2d)$，其中 $n$ 是序列长度，$d$ 是模型维度
- 内存使用随序列长度平方增长

### 训练技巧
- 学习率调度（预热 + 衰减）
- 梯度裁剪
- 标签平滑
- Dropout 正则化

## 代码示例

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 重塑为多头注意力
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # 连接多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.W_o(context)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 带残差连接的自注意力
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 带残差连接的前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

## 最新发展

### 高效 Transformer
- **稀疏注意力**：只关注位置的子集
- **线性注意力**：用线性复杂度近似注意力
- **Performer**：使用随机特征进行线性注意力

### 架构变体
- **GPT**：仅解码器用于自回归生成
- **BERT**：仅编码器用于双向理解
- **T5**：编码器-解码器用于文本到文本任务

## 参考文献

1. Vaswani, A., et al. "Attention is all you need." NIPS 2017.
2. Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019.
3. Radford, A., et al. "Language Models are Unsupervised Multitask Learners." OpenAI 2019.
