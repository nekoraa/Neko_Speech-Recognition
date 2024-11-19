# Neko_Speech-Recognition

基于 OpenAI **Whisper-large-v3-turbo** 的流式语音转文字系统，适用于游戏、会议、直播等场景。

---

## 功能特点

- **实时语音识别**：流式处理语音输入，实时生成字幕。
- **多语言支持**：可识别多种语言并输出对应的文本。
- **灵活适配**：支持游戏、在线会议、直播等多场景应用。
- **轻量部署**：优化的架构使其能够在普通硬件上高效运行。

---

## 环境依赖

本项目基于 Python 开发，推荐使用 **Python 3.8 或更高版本**。

### 必要依赖库

运行此项目需要以下 Python 库：

| **库名称**     | **用途**                                                                                     |
|----------------|---------------------------------------------------------------------------------------------|
| `torch`        | 深度学习框架，用于加载和运行 Whisper 模型。                                                   |
| `sounddevice`  | 音频设备接口，用于实时录音和播放。                                                            |
| `numpy`        | 数值计算库，用于处理音频数据。                                                               |
| `PyQt6`        | 构建用户界面的 GUI 框架。                                                                    |
| `colorama`     | 控制终端文本颜色，用于调试和信息提示。                                                        |
| `transformers` | Hugging Face 提供的模型库，用于加载 Whisper-large-v3-turbo 模型。                              |
| `googletrans`  | 谷歌翻译 API 的 Python 接口，用于实时翻译文本（可选）。                                        |
| `speechbrain`  | 用于音频处理的库，支持语音分离和增强（如 `SepformerSeparation`）。                              |

### 依赖安装

使用以下命令快速安装所需库：
```bash
pip install torch sounddevice numpy PyQt6 colorama transformers googletrans speechbrain



