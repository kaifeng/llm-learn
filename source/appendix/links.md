# 链接

## 教程

李沐实用机器学习，课程主页：https://c.d2l.ai/stanford-cs329p/

李沐动手学深度学习，课程主页：https://courses.d2l.ai/zh-v2
- 教材：https://zh-v2.d2l.ai
- 课程论坛讨论：https://discuss.d2l.ai/c/16
- Pytorch论坛：https://discuss.pytorch.org/

  [可选] 使用 conda/miniconda环境
  ```
  conda env remove d2l-zh
  conda create -y -n d2l-zh python-3.8 pip
  conda activate d2l-zh
  ```

  安装需要的包 `pip install -y jupyter d2l torch torchvision`

  下载代码并执行
  ```
  wget https://zh-v2.d2l.ai/d2l-zh.zip
  unzip d2l-zh.zip
  jupyter notebook
  ```

  > d2l需要python 3.8，用python3.12有不兼容（numpy安装失败）

  端口映射
  ssh -L8888:localhost:8888 ubuntu@remote-machine

吴恩达在线课程 https://www.deeplearning.ai/

李宏毅 2016机器学习 https://www.youtube.com/8zomhgKrsmQ

HuggingFace的LLM教程 https://huggingface.co/learn/llm-course

从头构建llama3模型 https://github.com/naklecha/llama3-from-scratch

Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks.
https://github.com/mlabonne/llm-course

numpy手写ml模型，来自普林斯顿的一位博士后将NumPy实现的所有机器学习模型全部开源，并提供了相应的论文和一些实现的测试效果。
https://github.com/ddbourgin/numpy-ml

「大模型」2小时完全从0训练26M的小参数GPT https://github.com/jingyaogong/minimind
此开源项目旨在完全从 0 开始，仅用 3 块钱成本 + 2小时，即可训练出仅为 25.8M 的超小语言模型MiniMind。
模型系列极其轻量，最小版本体积是 GPT-3 的 1/7000，力求做到最普通的个人GPU也可快速训练。
项目同时开源了大模型的极简结构-包含拓展共享混合专家(MoE)、数据集清洗、预训练(Pretrain)、监督微调(SFT)、LoRA微调， 直接偏好强化学习(DPO)算法、模型蒸馏算法等全过程代码。
MiniMind 同时拓展了视觉多模态的 VLM: MiniMind-V。
项目所有核心算法代码均从 0 使用 PyTorch 原生重构！不依赖第三方库提供的抽象接口。
这不仅是大语言模型的全阶段开源复现，也是一个入门 LLM 的教程。

扩散模型揭秘：从数学原理到代码实现，手把手教你生成高质量图像
https://www.youtube.com/watch?v=EhndHhIvWWw
深入解析了扩散模型的核心原理，重点解读了2020年里程碑论文《去噪扩散概率模型》(DDPM)。视频从数学角度拆解了扩散模型如何通过逐步添加和去除高斯噪声来生成数据：
1. 前向过程：通过固定噪声计划将图像逐渐转化为纯噪声
2. 反向过程：训练神经网络学习去噪，最终实现从噪声生成图像
3. 关键创新：DDPM简化了2015年原始扩散模型的训练目标，使其成为实用的生成技术
视频还包含完整的Python代码演示，展示如何：
- 在MNIST数据集上训练扩散模型
- 实现从纯噪声生成手写数字的采样过程
- 加载预训练模型生成逼真人脸图像
最后对比了扩散模型与VAE的生成质量差异，并预告了更高效的后续改进方法。

大模型相关技术原理以及实战经验 https://github.com/liguodongiot/llm-action

[spec-kit](https://github.com/github/spec-kit)，一个为 AI 驱动、规范驱动的开发而设计的开源工具包。它提供了一套用于定义、管理和执行开发任务规范的工具集，通过将开发流程分为规范、计划、任务和实施四个阶段，旨在帮助 AI Agent 更可靠、更准确地完成复杂工作。
