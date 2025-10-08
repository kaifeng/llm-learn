# AI应用

**AI画图**：Stable Diffusion, Midjourney, DALL-E 等工具，根据文本提示生成高质量图像。


## Ollama

**Ollama 是一个开源工具，旨在简化在本地机器上运行大型语言模型（LLM）的过程。** 它提供了一个统一的框架，让用户可以轻松地下载、安装、管理和运行各种开源 LLM（如 Llama 系列、Mistral、Gemma 等）。Ollama 抽象了模型部署的复杂性，包括模型量化、硬件加速（如 GPU 支持）和 API 服务，使得普通用户也能在自己的电脑上体验和开发基于 LLM 的应用。

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

## AI 绘画：Stable Diffusion 与 ComfyUI

### Stable Diffusion 模型

Stable Diffusion 是一个于 2022 年发布的、强大的文生图（Text-to-Image）AI 模型。它属于**潜在扩散模型（Latent Diffusion Model）** 的一种，其核心工作原理是在一个低维的潜在空间中对数据进行“加噪”和“去噪”处理，最终根据文本提示（Prompt）生成全新的、符合描述的图像。

与早期闭源的文生图模型（如 DALL-E 2）不同，Stable Diffusion 的最大特点是其**开源**属性。这使得研究人员和爱好者可以在消费级硬件上运行、微调和部署它，极大地推动了 AI 绘画技术的普及和创新生态的发展。

### ComfyUI：节点式工作流工具

ComfyUI 是一个为 Stable Diffusion 设计的、功能强大且高度模块化的图形用户界面。它最大的特点是其**基于节点（Node-based）的流程图式界面**。

用户可以通过拖拽代表不同功能的节点（如模型加载器、采样器、文本编码器等）并将它们连接起来，构建出完全自定义的图像生成工作流。这种方式虽然初看起来比简单的输入框更复杂，但它提供了无与伦比的灵活性和控制力，让用户能够：

- **精准控制流程**：清晰地看到数据（如模型、文本、图像）在每一步如何被处理。
- **构建复杂工作流**：轻松实现如图生图（Image-to-Image）、图像修复（Inpainting）、高清放大（Upscaling）等高级功能。
- **分享与复现**：整个工作流可以被保存为文件或图片，方便地与他人分享和复现，极大地促进了技巧的传播。