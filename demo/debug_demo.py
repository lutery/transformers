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
        intermediate_size=11008 // 2,
        num_hidden_layers=32 // 2,
        num_attention_heads=32 // 2,
        max_position_embeddings=2048 // 2,
    )

    llamaModel = LlamaModel(config=llamaConfig).to(deivce)

    input_ids = torch.randint(
        low=0, high=llamaConfig.vocab_size, size=(4, 30)).to(deivce)
    
    res = llamaModel(input_ids) # input_ids shape (batch_size, sequence_length)
    print(res)


if __name__ == "__main__":
    run_llama()