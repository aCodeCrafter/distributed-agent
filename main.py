from llama_cpp import Llama 

# 1. Import the model directly from a file path
llm = Llama(
    model_path="./models/Qwen3-0.6B-Q8_0.gguf",
    n_ctx=2048,  # Context window size
    n_gpu_layers=-1,  # Set to -1 to offload all layers to the GPU (if available)
)

while True:
    response = llm(
        input(">"),
        max_tokens=150,
        temperature=0.7,
    )
    print(response["choices"][0]["text"])
