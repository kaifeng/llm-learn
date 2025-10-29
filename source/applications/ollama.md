# Ollama

Ollama 是一个开源工具，旨在简化在本地机器上运行大型语言模型（LLM）的过程。它提供了一个统一的框架，让用户可以轻松地下载、安装、管理和运行各种开源 LLM（如 Llama 系列、Mistral、Gemma 等）。Ollama 抽象了模型部署的复杂性，包括模型量化、硬件加速（如 GPU 支持）和 API 服务，使得普通用户也能在自己的电脑上体验和开发基于 LLM 的应用。

ollama.com官网下载app

https://github.com/ollama/ollama/releases/latest/download/Ollama-darwin.zip

下载app后运行，按提示安装命令行工具，在命令行执行 `ollama run llama3.2` 下载模型。

```
>>>  /show info
  Model
    architecture        llama
    parameters          3.2b
    context length      131072
    embedding length    3072
    quantization        Q4_K_M

  Parameters
    stop    "<|start_header_id|>"
    stop    "<|end_header_id|>"
    stop    "<|eot_id|>"

  License
    LLAMA 3.2 COMMUNITY LICENSE AGREEMENT
    LLAMA 3.2 Version Release Date: September 25, 2024
```

然后就可以在Shell直接与模型交互。

执行ollama list查看大模型，run运行大模型
```
% ollama list
NAME               ID              SIZE      MODIFIED
llama3.2:latest    a80c4f17acd5    2.0 GB    10 minutes ago
% ollama run llama3.2
>>> Hi
Hello! How can I assist you today?

>>> Send a message (/? for help)
```
