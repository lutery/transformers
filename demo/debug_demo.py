from transformers.models.llama import LlamaModel, LlamaConfig
import torch

def select_device(cpu=False):
    if not cpu and torch.cuda.is_available():
        return torch.device("cuda")
    elif not cpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def run_llama():
    deivce = select_device()

    llamaConfig = LlamaConfig(
        vocab_size=32000, # 词表大小
        hidden_size=4096 // 2, # 隐藏层大小
        intermediate_size=11008 // 2, # 前馈网络（Feed-Forward Network, FFN）中间层的维度大小。作用：定义每个 Transformer 层中前馈网络的隐藏层大小，通常比 hidden_size 大
        num_hidden_layers=32 // 2, # Transformer 模型的隐藏层数量,32 // 2 = 16，表示模型包含 16 层 Transformer
        num_attention_heads=32 // 2, # 每个 Transformer 层中多头注意力机制的注意力头数量
        max_position_embeddings=2048 // 2, # 最大位置嵌入数，即模型支持的最大序列长度
    )

    llamaModel = LlamaModel(config=llamaConfig).to(deivce)

    input_ids = torch.randint(
        low=0, high=llamaConfig.vocab_size, size=(4, 30)).to(deivce)
    
    # input_ids shape (batch_size, sequence_length)
    # 4: batch_size, 30: sequence_length
    res = llamaModel(input_ids) # input_ids shape (batch_size, sequence_length)
    print(res)


if __name__ == "__main__":
    run_llama()