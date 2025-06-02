# 大语言模型部署项目

## 项目概述
本项目基于ModelScope平台，部署并测试了通义千问Qwen-1.8B-Chat和智谱ChatGLM3-6B两款中文大语言模型，通过对比分析它们在语义理解、逻辑推理等方面的表现。

## 环境配置
1. **平台准备**：
   - 注册ModelScope账号并绑定阿里云
   - 获取免费CPU计算资源

2. **基础环境**：
```bash
conda create -n llm_env python=3.10 -y
conda activate llm_env
pip install torch==2.3.0+cpu torchvision==0.18.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.33.3 modelscope==1.9.5 pydantic==1.10.13
```

## 模型下载
```bash
# 通义千问Qwen-1.8B-Chat
git clone https://www.modelscope.cn/qwen/Qwen-1_8B-Chat.git

# 智谱ChatGLM3-6B
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git
```

## 运行示例
### 通义千问Qwen-1.8B-Chat
```bash
python run_qwen_cpu.py
```

### 智谱ChatGLM3-6B
```bash
python run_chatglm_cpu.py
```

## 项目链接
- GitHub仓库: [Introduction-to-Artificial-Intelligence/hw4](https://github.com/SOLDIER-627/Introduction-to-Artificial-Intelligence/new/main/hw4)
