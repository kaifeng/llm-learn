# LLM领域概览

语言模型被训练为根据周围的词语上下文信息，预测一个词的概率，这使得模型对语言有基础的理解，可用于其它通用任务。

训练一个transformer模型有两种方式：
- Maked language modeling (MLM)：用于编码器模型（如BERT）。通过随机掩盖输入的某些词元，训练模型根据上下文预测原始的词元。
  这允许模型学习到双向的上下文。
- Causal language modeling (CLM)：用于解码器模型（如GPT）。基于过于的所有序列信息预测下一个词元。模型只能从之前的信息预测下一个词元。

原始的 Transformer 模型是一个包含编码器和解码器的完整架构，它在机器翻译等序列到序列（Seq2Seq）任务上取得了巨大成功。然而，根据任务性质的不同，只使用其中的一部分往往能达到更优的效果。这导致了 Transformer 架构的三个主要分支。
- Encoder-only模型：只使用 Transformer 的编码器部分。输入一个完整的句子，模型中的自注意力机制可以同时关注到句子中的所有词元，无视其前后位置。因此，它对整个句子的理解是全局的，非常适合需要对已有文本进行深入理解的任务，如句子分类、命名实体识别、问答等。代表模型为BERT及其所有变体（如 RoBERTa, ALBERT, DistilBERT）。
- Decoder-only模型：只使用 Transformer 的解码器部分。其核心是带掩码的自注意力。在处理或生成任何一个词元时，模型只能关注到它自己以及它之前的所有词元，无法看到未来的信息。这种从左到右的单向信息流使其本质上是一个自回归（Autoregressive）模型。适合于根据上文生成新文本的任务，如文本续写、故事创作、对话、代码生成等。代表模型为GPT系列（GPT-1, 2, 3, 4）、Llama 系列、Mistral 等。
- Encoder-decoder模型（如T5，BART）：保留了 Transformer 的完整结构。编码器首先将整个输入序列编码成一个富含信息的中间表示。然后，解码器在自回归地生成输出序列的每一步，都会通过“编码器-解码器注意力”来关注编码器产出的完整输入信息。适合需要将一个输入序列转换为另一个输出序列的任务，例如：机器翻译、文本摘要、问答等。代表模型有原始的 Transformer 模型、T5、BART。

## LLM的开发和应用

-   **预训练**：在海量无标签数据上进行自监督学习，使模型学习通用语言知识和世界模型。这是 LLM 智能的基石。相关技术有大规模分布式训练（如 Megatron-LM）、数据清洗与过滤。
-   **微调与对齐 (Fine-tuning & Alignment)**：
    -   **指令微调 (Instruction Fine-tuning)**：使模型学会遵循人类指令。
    -   **基于人类反馈的强化学习 (RLHF)**：进一步将模型行为与人类偏好和价值观对齐，提升有用性、无害性。
-   **部署与推理**：将训练好的模型投入实际使用，并优化其运行效率和成本。相关技术有高效推理引擎（如 vLLM）、模型量化、模型蒸馏。

## 常见模型类型

以下是一些常见的模型类型及其主要应用，Hugging Face模型库中有数万个开源模型可以提供下载。

### 自然语言处理 (NLP)

-   **文本分类 (Text Classification)**:
    -   **应用**: 情感分析、主题识别、垃圾邮件检测。
    -   **常见模型**: `BERT`, `RoBERTa`, `DistilBERT`。
-   **命名实体识别 (Token Classification / NER)**:
    -   **应用**: 从文本中识别人名、地名、组织机构等特定实体。
    -   **常见模型**: `BERT`, `ELECTRA`。
-   **问答 (Question Answering)**:
    -   **应用**: 根据给定的上下文回答问题（抽取式问答），或直接生成答案（生成式问答）。
    -   **常见模型**: `BERT`, `DistilBERT` (抽取式), `T5`, `BART` (生成式)。
-   **文本生成 (Text Generation)**:
    -   **应用**: 续写文本、创作故事、生成代码。
    -   **常见模型**: `GPT-2`, `BLOOM`, `Llama`。
-   **摘要 (Summarization)**:
    -   **应用**: 将长篇文章缩减为简短的摘要。
    -   **常见模型**: `BART`, `T5`, `Pegasus`。
-   **翻译 (Translation)**:
    -   **应用**: 将文本从一种语言翻译成另一种语言。
    -   **常见模型**: `T5`, `MarianMT`, `M2M100`。
-   **特征提取 (Feature Extraction)**:
    -   **应用**: 将文本转换为固定大小的向量（嵌入），用于语义搜索、聚类等下游任务。
    -   **常见模型**: `BERT`, `RoBERTa`, `Sentence-Transformers`。
-   **Mistral/Mixtral**:
    -   **应用**: 高效的文本生成，在性能和效率之间取得了很好的平衡。
    -   **常见模型**: `Mistral-7B`, `Mixtral-8x7B`
-   **Gemma**:
    -   **应用**: Google开发的轻量级开放模型，适用于各种规模的应用。
    -   **常见模型**: `gemma-2b`, `gemma-7b`
-   **Claude 3**:
    -   **应用**: 强调安全性和长上下文的对话和文本生成。
    -   **常见模型**: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`

### 计算机视觉 (Computer Vision)

-   **图像分类 (Image Classification)**:
    -   **应用**: 识别图像中的主要物体（例如，猫、狗、汽车）。
    -   **常见模型**: `ViT` (Vision Transformer), `ResNet`, `Swin Transformer`。
-   **目标检测 (Object Detection)**:
    -   **应用**: 在图像中定位多个物体并识别其类别。
    -   **常见模型**: `YOLOS`, `DETR`。
-   **图像分割 (Image Segmentation)**:
    -   **应用**: 对图像中的每个像素进行分类，以区分不同的物体和背景。
    -   **常见模型**: `SegFormer`, `Mask2Former`。

### 音频处理 (Audio)

-   **音频分类 (Audio Classification)**:
    -   **应用**: 识别音频中的声音事件（如掌声、警报声）或关键词。
    -   **常见模型**: `Wav2Vec2`, `HuBERT`。
-   **自动语音识别 (Automatic Speech Recognition, ASR)**:
    -   **应用**: 将语音转换为文字。
    -   **常见模型**: `Whisper`, `Wav2Vec2`。

### 多模态 (Multimodal)

这类模型能够同时处理多种信息模态（如文本和图像）。

-   **图文问答 (Visual Question Answering, VQA)**:
    -   **应用**: 根据图像内容回答相关问题。
    -   **常见模型**: `ViLT`, `BLIP`。
-   **图像描述生成 (Image Captioning)**:
    -   **应用**: 为图像生成描述性文字。
    -   **常见模型**: `BLIP`, `GIT`。
-   **Gemini**:
    -   **应用**: Google开发的原生多模态模型，能够理解和处理文本、图像、音频和视频。
    -   **常见模型**: `gemini-pro`, `gemini-ultra`