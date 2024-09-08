use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators::{self as OP, masked_softmax, matmul_transb, random_sample, rms_norm, silu,rope,gather};
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
pub struct Llama<T> {
    vocab: usize,           // vocab size 表示模型的词汇表大小，即模型能够处理的词汇数量
    //词汇表大小决定了模型可以理解和生成的不同 token（词汇、子词或字符）的数量。通常，对于 NLP 模型，词汇表包括所有可能的输入和输出词汇。

    n_layers: usize,        // number of layers 表示模型中的层数
    //Transformer 模型由多个层组成，每一层执行类似的操作。n_layers 决定了模型的深度，层数越多，模型的表达能力和复杂性也会增加。

    n_q_h: usize,           // number of heads for q 表示 Query 头的数量
    //Transformer 模型使用多头自注意力机制，n_q_h 表示 Query 向量的多头数量。每个头可以独立地关注输入序列的不同部分。

    n_kv_h: usize,          // number of heads for k and v 表示 Key 和 Value 头的数量
    //与 Query 类似，Key 和 Value 也有各自的多头机制。n_kv_h 指定了 Key 和 Value 的头数。
    //一般来说，n_kv_h 可以与 n_q_h 不同，以调整不同的注意力机制。

    d: usize,               // dimension of hidden states 隐藏状态的维度
    //d 决定了每层 Transformer 中间表示的维度，即每个 token 在处理过程中的特征向量的大小。较大的 d 提供了更多的表示能力。

    dqkv: usize,            // length of a single q, k, or v vector 单个 Query、Key 或 Value 向量的长度。
    //dqkv 决定了 Query、Key 和 Value 向量的维度大小。通常，它等于 d 除以多头的数量 (n_q_h)，因为每个头只分配到 d / n_q_h 的维度。

    di: usize,              // dimension of intermediate states 中间层的维度
    //在 Feed-Forward 网络中，隐藏层的大小通常比输入层和输出层的维度大，di 决定了这个中间层的维度。这个参数直接影响模型的复杂性。

    eps: f32,               // epsilon for RMS normalization RMS 归一化的 epsilon 值
    
    rope_theta: f32,        // rope theta for rope initialization RoPE初始化参数
    //LLaMA 模型中使用 RoPE 进行位置编码，rope_theta 是 RoPE 的一个参数，
    //用于控制位置嵌入的旋转频率。位置编码允许模型感知输入序列中各个 token 的相对位置。

    max_seq_len: usize,     // maximum sequence length 最大序列长度
    //表示模型能够处理的最大输入序列长度。输入序列的长度不能超过这个值，超出部分将被截断。

    params: LLamaParams<T>, // trained weights of this model 模型的训练权重参数
    //params 保存了模型中所有需要学习的参数，包括网络中的权重矩阵（如 Attention 权重、Feed-Forward 网络权重等）。
    //LLamaParams<T> 是另一个结构体，包含模型的所有权重。

    bos_token_id: u32,      // start token id 序列的起始 Token ID
    //表示序列的起始符号的 token ID，用于生成文本或进行推理时，告诉模型这是序列的开始。通常在文本生成任务中使用。

    eos_token_id: u32,      // end token id 序列的结束 Token ID
    //表示序列的结束符号的 token ID，用于告诉模型生成结束。这在自动生成文本时非常重要，模型在生成到该 token 时会停止继续生成
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut x = Tensor::<f32>::default(&vec![seq_len, self.d]);

        // Computation Starts Here
        // Embedding lookup
        gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            self_attention(
                &mut x,
                &mut att_scores,
                q,
                &full_k,
                &full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );

            // x = x @ O_weight.T
            matmul_transb(&mut hidden_states, 0., &x, &self.params.wo[layer], 1.0);

            // residual = x + residual
            let len = residual.size();
            assert!(len == hidden_states.size());
            let _r = unsafe { residual.data_mut() };
            let _h = hidden_states.data();
            for i in 0..len {
                _r[i] += _h[i];
            }

            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        let mut result = Vec::<u32>::new();
        let mut cache = self.new_cache();
        let mut token: Vec<u32> = Vec::from(token_ids);
        if token[0] != self.bos_token_id {
            token.insert(0, self.bos_token_id);
        }
        let mut input = Tensor::<u32>::new(token, &vec![1, token_ids.len()]);
        loop {
            let output =
                random_sample(&self.forward(&input, &mut cache), top_p, top_k, temperature);
            result.push(output);
            if result.len() >= max_len || output == self.eos_token_id {
                break;
            }
            input = Tensor::<u32>::new(Vec::from([output]), &vec![1, 1]);
        }

        result
    }
}

fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv) 表示输入序列的隐藏状态张量，用于存储注意力计算后的输出结果。
    //seq：序列长度  n_kv_h：Key 和 Value 的头数量  n_groups：头分组的数量 dqkv：单个 Query、Key 或 Value 向量的维度

    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq) 存储自注意力的得分
    //n_kv_h：Key 和 Value 头数量  n_groups：头分组的数量  seq：当前输入序列的长度  total_seq：总序列长度，可能包含缓存的过往序列

    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv) Query 张量，表示查询向量
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv) Key 张量，表示键向量，用于与 Query 进行点积计算注意力得分
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv) Value 张量，表示值向量，用于根据注意力得分加权得到输出
    n_kv_h: usize,                   //Key 和 Value 的头数  每个头是注意力机制的一个独立实例，可以学习不同的注意力模式。
    n_groups: usize,                 //头的分组数量。在某些模型中，头被划分为不同的组以进行并行计算。
    seq_len: usize,                  //输入序列的长度，即本次前向传播中的序列长度。
    total_seq_len: usize,            //总序列长度，通常包括过去的序列长度（在缓存中），因此等于 past_seq_len + seq_len。
    dqkv: usize,                     //单个 Query、Key 或 Value 向量的维度。
) {
    // score = Q @ K.T / sqrt(dim)
    let _a = unsafe { att_scores.data_mut() };
    let _q = q.data();
    let _k = k.data();
    let _v = v.data();

    //self-attention的核心步骤计算query和key的点积
    let sqrt = (dqkv as f32).sqrt();
    for h in 0..n_kv_h * n_groups { //遍历头
        for l in 0..seq_len {  //遍历序列位置
            for i in 0..total_seq_len {  //每个序列位置上对应的token
                let sum = (0..dqkv)  //对 dqkv 范围内的每个元素进行遍历计算。
                    .map(|j| {  //对于 j（从 0 到 dqkv-1 的每个值），进行 Query 和 Key 对应维度的点乘。
                        _q[l * n_kv_h * n_groups * dqkv + h * dqkv + j]  //获取第 l 个位置、第 h 个头的第 j 个维度的 Query 向量元素。
                            * _k[i * n_kv_h * dqkv + h / n_groups * dqkv + j]  //获取第 i 个位置、第 h 个头的第 j 个维度的 Key 向量元素。
                    })
                    .sum::<f32>();//对所有维度的 Query 和 Key 元素的点积进行累加，得到 Query 和 Key 向量之间的相似度分数（即点积的和）。
                _a[h * seq_len * total_seq_len + l * total_seq_len + i] = sum / sqrt;
                //将计算出的相似度分数除以 sqrt（即 dqkv 的平方根）以标准化相似度分数，防止数值过大导致训练不稳定。
                //将归一化后的相似度分数存储到 att_scores 张量中的正确位置。这个位置对应第 h 个头，
                //第 l 个 Query 对应第 i 个 Key 的相似度分数。
            }
        }
    }

    // attn = softmax(score)
    masked_softmax(att_scores);
    //将得到的自注意力得分通过 Softmax 转化为概率分布，表示每个序列位置对其他序列位置的注意力权重。
    //这一步操作使得得分变为一个在 [0, 1] 之间的概率值，所有得分的总和为 1，表示当前 token 对每个位置的关注程度。

    // x = attn @ V
    let _a = att_scores.data();
    let _h = unsafe { hidden_states.data_mut() };
    for h in 0..n_kv_h * n_groups {
        for l in 0..seq_len {
            for i in 0..dqkv {
                let sum = (0..total_seq_len)//计算当前 Query 对应的加权 Value。
                    .map(|j| {
                        _a[h * seq_len * total_seq_len + l * total_seq_len + j]//取出 att_scores 张量中第 h 个头，第 l 个 Query 对应第 j 个 Key 的注意力分数。
                            * _v[i + h / n_groups * dqkv + j * n_kv_h * dqkv]//取出 Value 张量中第 j 个 Key 的 Value 元素（对应第 i 维的向量）。
                    })
                    .sum::<f32>();//对所有位置 j 的加权 Value 求和，得到当前 Query 对所有 Key 的加权和，表示最终的输出。
                _h[l * n_kv_h * n_groups * dqkv + h * dqkv + i] = sum;//将加权求和的结果存储到 hidden_states 张量中，表示对 Query 的计算结果
            }
        }
    }
}


fn mlp(
    residual: &mut Tensor<f32>,      // 4 2
    hidden_states: &mut Tensor<f32>, //4 2
    gate: &mut Tensor<f32>,          //4 3
    up: &mut Tensor<f32>,            // 4,3
    w_up: &Tensor<f32>,              // 3,2
    w_down: &Tensor<f32>,            // 2.3
    w_gate: &Tensor<f32>,            // 3,2
    rms_w: &Tensor<f32>,             // 2
    eps: f32,
) {
    // 1. 对 residual 进行 RMS 归一化，结果存储在 hidden_states 中
    //    hidden_states = rms_norm(residual)
    rms_norm(hidden_states, residual, rms_w, eps);

    // 2. 计算 gate = hidden_states @ w_gate.T
    //    gate 用于后续的激活函数计算
    matmul_transb(gate, 0.0, hidden_states, w_gate, 1.0);

    // 3. 计算 up = hidden_states @ w_up.T
    //    up 是另一个中间结果，形状与 gate 相同
    matmul_transb(up, 0.0, hidden_states, w_up, 1.0);

    // 4. 通过 SiLU 激活函数计算 gate = gate * sigmoid(gate) * up
    //    结果存储在 up 中，以便下一步使用
    //    实际上, 这里将计算 gate * sigmoid(gate) * up 存储在 up 中
    silu(up, &gate);

    // 5. 计算 residual = residual + up @ w_down.T
    //    这一步同时计算了 hidden = up @ w_down.T 并将其加到 residual 中
    //    通过设置 beta = 1.0，完成了 residual = hidden + residual 的操作
    matmul_transb(residual, 1.0, &up, w_down, 1.0);
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );
    residual.print();
    residual.print();
    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}