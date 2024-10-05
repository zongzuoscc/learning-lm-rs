# 简单大模型推理系统

本项目使用rust编写完成了一个简单的大模型推理程序
由于本人能力与时间精力有限，处在大二阶段且刚接触生成式ai的学习，只能通过不到两个月的时间完成文本生成功能

通过此次项目也提升了git的使用技巧，并且学习了一门新的语言rust

在此次项目中，不止学习了infinitensor训练营中的课程，也参考了以下课程

[从零开始构建 GPT 系列](https://www.bilibili.com/video/BV11yHXeuE9d/?p=7&share_source=copy_web&vd_source=44a0f2675d4e61f83d2eacdcba85bcf7&t=3720)

[动手学习深度学习视频讲解](https://www.bilibili.com/video/BV1if4y147hS?vd_source=02f908c3673a022f7ceb9b5fb1347e23)

[动手学深度学习课本](https://zh.d2l.ai/)


如何运行生成story项目
```bash
  cd learning-lm-rs
  cargo run
```
以下为生成的故事效果

<center>  <!--开始居中对齐-->

(![● README.md - exam-grading - Visual Studio Code 2024_9_26 21_50_58.png](https://github.com/zongzuoscc/learning-lm-rs/blob/main/%E2%97%8F%20README.md%20-%20exam-grading%20-%20Visual%20Studio%20Code%202024_9_26%2021_50_58.png))
</center> <!--结束居中对齐-->

--- 

### 神经网络
神经网络通常由以下几种类型的层组成：

输入层：接收原始数据作为网络的输入。
隐藏层：网络中的中间层，可以有多个，负责对输入数据进行处理和特征提取。
输出层：产生网络的最终输出，其神经元的数量和类型取决于特定的任务（如分类、回归等）。

### silu算子(激活函数)

SILU 是一种激活函数，用于在神经网络的隐藏层中引入非线性，帮助模型学习更复杂的特征表示。SILU 作为激活函数，通常应用于神经网络的隐藏层，特别是在需要动态调整信息流的场景中。

1.平滑激活：输出值在-1到1之间，但与sigmoid函数不同，当输入过大或很小时，silu函数的输出不会饱和

2.数据规范化：将数据压缩到一个特定的范围可以防止激活值过大或过小，从而有助于防止梯度消失或梯度爆炸问题

3.非线性映射，平滑性和连续性，减少过拟合，提高数值稳定性（尤其在浮点数运算时），与人类感知的相似性

$$
y=silu(x) × y
$$

$$
silu(x) = sigmoid(x) × x
$$


$$
sigmoid(x) = \frac{1}{1 + e^{-x}}
$$

依照上述公式即可实现silu函数


### RMS Normalization 
RMS Normalization 是一种输入数据预处理或层间规范化的技术，主要目的是调整数据的尺度，使模型训练更加稳定。通过对数据进行规范化来减少内部协变量偏移，加速模型的收敛，提高模型的性能和泛化能力。

RMS Normalization 通常应用于模型的输入层或中间层，对输入数据或中间特征进行规范化处理。

在编写rms_norm函数时，确保y至少有两个维度

通过对每个向量的平方和进行缩放。这样可以确保网络各层之间的信号不太强也不太弱。


$$
y_i=\frac{w×x_i}{\sqrt{ \frac{1}{n} \sum_{j} x_{ij}^2 +\epsilon}}
$$


### feed-forward（mlp）函数
Feed-forward（MLP）函数通常指的是前馈神经网络中的多层感知机（Multilayer Perceptron）模型。在机器学习和深度学习领域，MLP 是一种基本的神经网络结构，它由多个层组成，每层包含若干个神经元（或节点）。MLP 的特点是信息只在一个方向上流动，从输入层到隐藏层，再到输出层，层与层之间是全连接的。

##### 前馈神经网络
前馈神经网络（Feedforward Neural Network，FNN）是人工神经网络的一种，它由输入层、隐藏层和输出层组成，各层之间是全连接的，但同一层内的神经元之间没有连接。前馈神经网络的信号从输入层开始，只向前传播，经过一个或多个隐藏层，最终到达输出层，而不会在网络内部形成反馈循环。

计算过程如下

``` python
hidden = rms_norm(residual)
gate = hidden @ gate_weight.T
up = hidden @ up_weight.T
hidden = gate * sigmoid(gate) * up ## silu
hidden = hidden @ down_weight.T
residual = hidden + residual
```

### word embedding
词嵌入（Word Embeddings）：如 Word2Vec、GloVe、FastText 等，用于将单词映射到向量。

### encoder decoder

Encoder 是一个将输入数据转换为另一种形式或表示的模块，通常用于提取数据的特征或压缩数据。在 NLP 中，encoder 通常用于将文本转换为固定长度的向量表示，这些表示捕捉了输入文本的关键信息。

Encoder 的一些应用包括：

序列到向量：将文本序列转换为固定大小的向量。
特征提取：从原始数据中提取有用的特征。
数据压缩：减少数据的维度，同时保留最重要的信息。

Decoder 是一个将编码后的数据或表示转换回原始形式或另一种有用的形式的模块。在 NLP 中，decoder 通常用于生成文本，如在机器翻译或文本摘要任务中将编码后的向量转换回文本。

Decoder 的一些应用包括：

序列生成：从固定大小的向量生成文本序列。
数据重建：从压缩的表示重建原始数据。
翻译：将一种语言的文本转换为另一种语言的文本。

### self-attention


Self-attention 是现代自然语言处理模型中最核心的组件，使模型能够为输入序列的每一个 token 分配一个权重，这个权重基于该 token 和序列中其他所有 token 的关系。这样，模型可以学习到哪个 token 在理解当前 token 时更重要。

---

### Self-Attention 步骤解析

#### 1. **`x = rms_norm(residual)`**  
   **操作**：对 `residual` 进行 **RMS Normalization**。
   
   **意义**：  
   - `residual` 是当前输入序列的表示，每个元素可能包含输入 token 的嵌入表示或前一层的输出。通过 RMS Normalization 可以确保输入具有一致的规模，避免输入的数值不平衡导致模型不稳定。
   - RMS Normalization 是一种归一化方法，它通过对每个向量的平方和进行缩放。这样可以确保网络各层之间的信号不太强也不太弱。

---

#### 2. **`Q = RoPE(x @ Q_weight.T)`**  
   **操作**：`x` 与 **Q_weight.T**（Query 权重矩阵的转置）做矩阵乘法，再应用 **RoPE（Rotary Positional Encoding）**。
   
   **意义**：
   - **Query（Q）** 是从 `x` 生成的，用于表示 "查询" 信息，后续会与 Key 一起计算注意力得分。
   - **RoPE** 是一种位置编码方法，用于将序列中的位置信息编码到 `Q` 中，从而使模型能够理解 token 之间的位置关系（这对自注意力非常重要，因为它本身不感知位置）。
   - **x @ Q_weight.T** 是通过乘以权重矩阵生成 `Q`。权重矩阵可以看作模型通过学习后获取的某种"查询映射"。
   - 每个 `Q` 向量的维度通常是输入向量的维度除以头的数量（即 `dqkv`）。

---

#### 3. **`K = RoPE(x @ K_weight.T)`**  
   **操作**：`x` 与 **K_weight.T**（Key 权重矩阵的转置）做矩阵乘法，再应用 **RoPE**。
   
   **意义**：
   - **Key（K）** 是与 `Q` 相对应的，表示“键”的信息。接下来 `Q` 和 `K` 会通过点积操作来计算注意力得分。
   - 类似于 `Q`，`K` 也通过乘以权重矩阵来获得，并且应用了 **RoPE** 进行位置编码，以捕获序列中的位置信息。
   - `Q` 和 `K` 的维度通常是相同的，这样后续的点积计算可以顺利进行。

---

#### 4. **`V = x @ V_weight.T`**  
   **操作**：`x` 与 **V_weight.T**（Value 权重矩阵的转置）做矩阵乘法。
   
   **意义**：
   - **Value（V）** 表示输入序列的实际信息，接下来它将通过注意力得分进行加权。  
   - **Value** 主要是用来存储最终输出的各个部分，而 `Q` 和 `K` 是用来计算注意力权重的。
   - 与 `Q` 和 `K` 不同，`V` 不需要应用 RoPE，因为它不参与位置相关的注意力计算，只是存储结果。

---

#### 5. **`K = cat(K_cache, K)`**  
   **操作**：将 `K_cache`（之前的 `Key` 缓存）与当前的 `K` 进行拼接。
   
   **意义**：
   - **拼接缓存**：在生成任务中，当前输入的 `K` 需要与之前的 `K` 拼接，这样可以保持上下文。生成模型常常会使用 **cache** 来保存以前计算的 `K` 和 `V`，以便高效地处理长序列。
   - 这一步的意义在于，处理长序列时不用重新计算先前已经计算过的 `K`，提高了生成效率。

---

#### 6. **`V = cat(V_cache, V)`**  
   **操作**：同理，将 `V_cache` 与当前的 `V` 进行拼接。
   
   **意义**：与 `K` 类似，`V` 也需要与缓存中的 `V` 拼接，保留之前的上下文信息。用于后续生成序列时参考之前的值。

---

#### 7. **`score = Q @ K.T / sqrt(dim)`**  
   **操作**：通过 `Q` 和 `K` 的点积计算注意力得分（score），并进行归一化。
   
   **意义**：
   - **点积**：`Q` 和 `K` 的点积表示查询和键之间的相似度，表示一个 token 关注另一个 token 的程度。  
   - **`sqrt(dim)`**：通过 `dim`（即 `dqkv`）的平方根来归一化，这可以避免数值过大。`dqkv` 是 `Q`、`K`、`V` 向量的维度。
   - **得分矩阵**：最终得到的是一个 `score` 矩阵，每个值表示 Query 对某个 Key 的注意力强度。

---

#### 8. **`attn = softmax(score)`**  
   **操作**：将 `score` 经过 **Softmax** 变换，使得每行的元素和为 1。
   
   **意义**：
   - **softmax** 操作将每个 Query 对 Key 的点积结果转化为概率分布，表示该 Query 关注不同 Key 的强度比例。每行的和为 1。
   - 通过 softmax 操作，可以确保注意力得分是一个合理的概率分布，而不是一组无约束的数值。

---

#### 9. **`x = attn @ V`**  
   **操作**：通过 `attn` 和 `V` 的矩阵乘法，计算加权后的 `V` 值。
   
   **意义**：
   - **attn** 表示每个 Query 对 Key 的注意力权重，`V` 存储的是实际的值。
   - **矩阵乘法**：将 `attn` 矩阵与 `V` 矩阵相乘，相当于为 `V` 加权，结果就是 `x`，即每个 Query 对不同 Key 的加权平均。
   - 这一步的输出是每个 token 通过注意力机制得到的最终表示，包含了整个输入序列的全局信息。

---

#### 10. **`x = x @ O_weight.T`**  
   **操作**：将 `x` 通过 **O_weight**（输出权重矩阵）变换，进行线性映射。
   
   **意义**：
   - **输出权重**：`O_weight` 是一个模型学习到的权重矩阵，用来将 `x` 进行进一步的线性映射，生成更复杂的表示。  
   - 这一步将 `x` 转换为符合网络下一步输入要求的形式。

---

#### 11. **`residual = x + residual`**  
   **操作**：将 `x` 与原始的 `residual` 做 **残差连接**。
   
   **意义**：
   - **残差连接**：这种操作确保模型的每一层都能“跳跃”地连接回去，并直接参考原始输入。  
   - 通过残差连接，可以解决深度网络中的梯度消失问题，使得信息能够直接从前面的层传播到后面的层，提升模型的训练效果。

---

### 总结

Self-attention 机制的核心是计算每个 token 与序列中其他 token 之间的注意力权重，利用这些权重来对 `Value` 加权，最终输出一个新的表示。每个 token 的表示都会根据其与其他 token 的关系进行更新，使得模型能够捕获长距离的依赖关系。

Self-attention 的每一步都环环相扣，依次完成了以下任务：
1. 通过 `Q`、`K`、`V` 矩阵生成注意力机制所需的基础信息。
2. 使用 `Q` 和 `K` 计算注意力得分，进行归一化。
3. 使用 softmax 将注意力得分转化为概率分布。
4. 根据注意力得分对 `V` 进行加权平均。
5. 通过线性变换和残差连接，得到每一层的输出表示。

这样，模型不仅能够理解序列中的局部信息，还能够结合全局上下文信息，从而在生成或理解任务中表现得非常强大。

需要注意的是，最后两步，即 **`x = x @ O_weight.T`** 和 **`residual = x + residual`**  是在函数之外单独实现的

### generate
生成式 AI 模型生成文本的核心原理是基于语言模型的概率预测。模型根据输入的上下文信息，预测出下一个 token（词或字符）。这是通过计算每个可能的 token 的概率分布来实现的。生成过程通过反复选择最可能的下一个 token 来构建句子或段落，直到达到指定的长度或遇到结束 token（如 <eos>）。

#### generate函数的流程
生成流程
1. 初始化：token_ids 是输入的上下文，可能是文本的一部分，或是前面的生成内容。我们将其转化为一个向量，并确保有开始 token。
2. 缓存：生成的模型需要记住之前的 Key 和 Value，因此我们使用 cache 来保存它们，避免重复计算。
3. 生成循环：
    - 前向传播：通过 forward 函数计算 logits，logits 是对下一个 token 的概率分布。
    - 采样：通过 random_sample 函数，使用 Top-k 或 Top-p 采样策略，从 logits 中选出下一个 token。温度参数 (temperature) 控制生成的随机性。
    - 生成终止条件：当生成的 token 达到最大长度或遇到结束 token (eos_token_id) 时，停止生成。
4. 返回结果：最终返回生成的 token 序列。


#### forward函数作用  

 **Embedding Lookup** ：通过词嵌入层将输入的 token ids 转化为向量表示。
 **Attention 计算** ：在每一层中，模型计算 Query、Key 和 Value，然后通过 self-attention 得到新的 token 表示。
 **残差连接** ：将经过注意力机制后的 token 表示与原始 token 表示相加，增强信息传递。
 **MLP 层** ：进一步处理每个 token 的表示，得到最终的输出。
 **最终输出 logits** ：经过 MLP 和输出层的计算，得到每个 token 的概率分布 logits，用于预测下一个 token。

--- 
欢迎各位同学。本课程中，各位将用Rust语言分阶段实现一个简单的大模型推理程序。

本课程分为两个阶段：作业阶段，各位将实现大模型的几个关键算子，Feed-Forward神经网络，以及大模型的参数加载；项目阶段，各位将实现大模型最为核心的Self-Attention结构，完成大模型的文本生成功能。之后，可以选择继续实现AI对话功能，搭建一个小型的聊天机器人服务。

- 本项目支持Llama、Mistral及其同结构的Transformer模型，所使用的数据类型为FP32，使用CPU进行推理。当然，欢迎各位同学在此基础上进行拓展。
- 本项目使用safetensors模型格式，初始代码只支持单个文件的模型。
- 本项目自带两个微型的语言模型，分别用于文本生成和AI对话（模型来自于Hugginface上的raincandy-u/TinyStories-656K和Felladrin/Minueza-32M-UltraChat）。对话模型比较大，需要到github页面的release里下载。

## 一、作业阶段

### 作业说明

- 你的代码需要通过全部已有的测试才能晋级下一阶段（项目包含github on-push自动检测）。
- 请在指定文件中和位置添加你的代码，不要修改其他文件和函数、文件名称和项目结构。作业阶段不需要额外的第三方依赖。
- 请不要修改已有的测试代码。开发过程中如果有需要，你可以添加自己的测例。
- 调试代码时，你可以打印张量的数据，你也可以使用pytorch中的函数辅助调试。

### 1. 算子：SiLU函数（10分）

请在`src/operators.rs`中实现SiLU算子，其公式为：

$$
y=silu(x) × y
$$

其中

$$
silu(x) = sigmoid(x) × x
$$

$$
sigmoid(x) = \frac{1}{1 + e^{-x}}
$$

注意：

- $`y`$ 既是输入，也存储最终输出。

- 该算子是element-wise操作而非向量点乘，即单次运算只涉及输入和输出张量中对应的元素。

- 你可以默认输入输出长度相同，不用考虑广播的情况。

- 用`src/operators.rs`中的测例检验你的实现是否正确。

### 2. 算子：RMS Normalization（20分）

请在`src/operators.rs`中实现RMS Normalization，其公式为：

$$
y_i=\frac{w×x_i}{\sqrt{ \frac{1}{n} \sum_{j} x_{ij}^2 +\epsilon}}
$$

注意：

- 你可以只考虑对最后一维进行计算的情况。即张量 $X(...,n)$ 和 $Y(...,n)$ 都是由若干个长度为 $n$ 的向量 $x_i, y_i$ 组成的，每次求和都在向量内进行。参数 $`w`$ 是个一维向量，与各个向量长度相同，且进行element-wise乘法。

- 用`src/operators.rs`中的测例检验你的实现是否正确。

### 3. 算子：矩阵乘（30分）

想必前两个算子的实现中你已经充分热身，那么重量级的来了。请在`src/operators.rs`中实现矩阵乘（Transpose B）算子，其公式为：

$$
C=\alpha AB^T + \beta C
$$

你有充足的理由质疑为什么这个矩阵乘算子要长成这个样子，以及为什么不用线代课上学的 $C=AB$ 这样更简洁的形式。

首先，我们为什么要对B矩阵进行转置？这其实涉及到机器学习中线性（linear）层的定义习惯，$`y = xW^T + b`$ 。矩阵乘算子中的 $`B`$ 矩阵常常是权重矩阵，它的每一行是一个与输入$`x`$中每一行向量等长的权重向量，而行的数量则对应了特征数。我们使用的模型的参数，也是按照这个方法存储的。

其次，为什么要加 $`C`$ 矩阵？大家如果查阅BLAS中矩阵乘的标准定义就会发现它也有这一项操作，而我们这里其实实现的就是BLAS矩阵乘的一个简化版本。将矩阵乘结果加到原本的矩阵上在实际中是应用很广泛的。比如线性层的bias，尽管这次我们实现的Llama模型并没有使用bias。之后实现全连接网络时，你就会发现有了这一项，我们可以和矩阵乘一起实现残差连接（residual connection）的功能。如果你只想计算 $`C=AB^T`$ 那么你可以在传参时将 $`\beta`$ 参数设置为0，将 $`\alpha`$ 参数设置为1。

你可以默认输入输出都是二维矩阵，即 $`A`$ 形状为 $`m×k`$，$`B`$ 形状为 $`n×k`$，$`C`$ 形状为 $`m×n`$，可以不用考虑广播的情况。到了项目部分，你有可能需要（非必须）实现支持广播的矩阵乘，比如$`(b, h, m, k) · (b, 1, k, n)`$ 这种情况，你可以去pytorch官方文档看到关于broadcast的规则。

你可以用`src/operators.rs`中的测例检验你的实现是否正确。

### 4. 模型结构：Feed-Forward神经网络（20分）

请在`src/models.rs`中实现Feed-Forward神经网络（mlp函数），计算过程如下：

``` python
hidden = rms_norm(residual)
gate = hidden @ gate_weight.T
up = hidden @ up_weight.T
hidden = gate * sigmoid(gate) * up ## silu
hidden = hidden @ down_weight.T
residual = hidden + residual
```

如果你正确地实现了之前地几个算子，那么这个函数的实现应该是相当简单的。需要注意的是，上一层的输出存储于residual这个临时张量中，这就是用到了我们之前提到的残差连接的概念，最终我们实现的神经网络的输出也要加上前一层的residual并存储于residual中，以便于下一层的计算。hidden_states则用于存储过程中的计算结果。你可以用`src/model.rs`中的测例检验你的实现是否正确。

### 5. Llama模型参数加载（20分）

请结合课上所讲的模型结构，根据代码种的定义在`src/params.rs`以及`src/model.rs`中补全大模型参数加载代码。项目已经为你做好了safetensors以及json文件的读取功能，你需要将参数原始数据以代码中的形式存于正确的位置，并赋予模型对象正确的config属性。safetensors里带有各张量的名称，应该足够你判断出张量代表的是哪个参数。

以下是大模型config中一些比较重要的属性的含义：

```python
{
  "bos_token_id": 1, # 起始符token id
  "eos_token_id": 2, # 结束符token id
  "hidden_size": 128, # 隐藏层大小，即各层输出的最后一维
  "intermediate_size": 384, # Feed-Forward神经网络的中间层大小
  "max_position_embeddings": 512, # 最大序列长度
  "num_attention_heads": 8, # Self-Attention的Q头数
  "num_hidden_layers": 2, # 隐藏层数
  "num_key_value_heads": 4, # Self-Attention的K和V头数
  "rms_norm_eps": 1e-6, # RMS Normalization的epsilon参数
  "rope_theta": 10000.0, # RoPE的theta参数
  "tie_word_embeddings": true, # 起始和结束embedding参数矩阵是否共享同一份数据
  "torch_dtype": "float32", # 模型数据类型
  "vocab_size": 2048 # 词表大小
}
```

注意：

- safetensors里存储的是原始数据，你需要以FP32的形式读取出来，创建出项目所使用的张量。
- safetensors包含张量的形状，你无需对原始张量做任何变形。
- 当"tie_word_embeddings"属性被打开时，模型最开始以及最后的embedding矩阵数据相同，safetensors会只存储一份数据，我们测试用的story模型就是这样。作业阶段你可以只关心story模型，但是后续项目中你需要处理两个矩阵不同的情况。
- 你可以用`src/model.rs`中的测例检验你的实现是否正确。

## 二、项目阶段

### 1. 模型结构：Self-Attention

恭喜你，来到了本项目最为核心的部分。在开始写代码前，建议你对着课上讲的大模型结构图把每一次计算所涉及的张量形状都推导一遍，尤其是对于“多头”的理解。项目已经帮你实现了kvcache的部分和RoPE等一些算子，写这些代码其实对于大模型的学习很有帮助，但是为了不让项目过于新手不友好而省略了。

在输入经过三个矩阵乘后，我们分别得到了Q、K、V三个张量，其中Q的形状为 (seq_len, q_head×dim) ，而K、V在连接完kvcache后的形状为 (total_seq_len, k_head×dim)，其中seq_len是输入序列的长度，可以大于1，total_seq_len是输入序列和kvcache的总长度 。你应该还记得课上的内容，在Q和K进行矩阵乘后，我们希望对于seq_len中的每个token每个独立的“头”都得到一个 (seq_len, total_seq_len) 的权重矩阵。这里就出现了两个问题:

第一，Q的头数和KV的头数并不一定相等，而是满足倍数关系，一般Q头数是KV头数的整数倍；假如Q的头数是32而KV头数是8，那么每4个连续的Q头用一个KV头对应。

第二，我们需要将 (seq_len, dim) 和 (dim, total_seq_len) 的两个矩阵做矩阵乘才能得到我们想要的形状，而现在的QK都不满足这个条件；你有几种不同的选择处理这个情况，一是对矩阵进行reshape和转置（意味着拷贝），再用一个支持广播（因为你需要对“头”进行正确对应）的矩阵乘进行计算，二是将这些矩阵视为多个向量，并按照正确的对应关系手动进行索引和向量乘法，这里我推荐使用更容易理解的后一种方法。

同样的，在对权重矩阵进行完softmax后和V进行矩阵乘时也会遇到这个情况。

对于每个头，完整的Self-Attention层的计算过程如下；

``` python
x = rms_norm(residual)
Q = RoPE(x @ Q_weight.T)
K = RoPE(x @ K_weight.T)
V = x @ V_weight.T
K = cat(K_cache, K)
V = cat(V_cache, V)
### 以下是你需要实现的部分
score = Q @ K.T / sqrt(dim)
attn = softmax(score)
x = attn @ V
x = x @ O_weight.T
residual = x + residual
```

Self-Attention的调试是很困难的。这里推荐大家使用pytorch来辅助调试。各位可以用transformers库（使用llama模型代码）来加载模型并运行，逐层检查中间张量结果。

### 2. 功能：文本生成

请在`src/model.rs`中补充forward函数的空白部分，实现generate函数。注意在foward函数的准备阶段，我们定义了几个计算用的临时张量，这是为了在多层计算中不重复分配内存，这些临时张量会作为算子函数调用的参数，你可以根据自己的需要更改这一部分（你其实可以用比这更小的空间）。

文本生成所需的采样的算子已为你写好。你需要初始化一个会被复用的kvcache，并写一个多轮推理的循环，每一轮的输出作为下一轮的输入。你需要根据用户传的最大生成token数以及是否出现结束符来判断是否停止推理，并返回完整推理结果。

所使用的模型在`models/story`中。`src/main.rs`已经为你写好了tokenizer的编码和解码，代码完成后，可以直接执行main函数。

### 3. （可选）功能：AI对话

仿照文本生成的功能，写一个实现AI对话的chat函数，之后你可以搭建一个支持用户输入的命令行应用。你需要在多轮对话中，保存和管理用户的kvcache。

你可以使用`models/chat`中的对话模型。其对话模板如下：

``` text
"{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
```

这种模板语言叫做Jinja2，在本项目中你可以不用实现任意模板的render功能，直接在代码中内置这个模板。你可以忽略system角色功能。下面是一个首轮输入的例子：

``` text
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
```

后续每轮输入也都应该使用该模板。如果你忘记了如何使用模板生成正确的输入，请回顾课堂上讲到的内容，提示：我们的模型的基础功能是故事续写。

如果你完成了项目，请向导师展示你的成果吧！其实这个项目还有很多可以拓展的地方，比如其他数据类型的支持、多会话的支持、GPU加速等等，欢迎你继续探索。
