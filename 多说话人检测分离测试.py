import os

import torch
from speechbrain.inference import SepformerSeparation as separator
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import os
import sys
import re
import torch
import sounddevice as sd
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt, QPoint
from PyQt6.QtGui import QTextCursor, QIcon
from colorama import Fore
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import threading

import warnings
import sys
from PyQt6.QtWidgets import QApplication, QComboBox, QPushButton, QVBoxLayout, QWidget, QTextEdit, QHBoxLayout, QLabel, \
    QSlider, QToolButton
warnings.filterwarnings("ignore")
from googletrans import Translator


# 加载预训练模型
model = separator.from_hparams(source="speechbrain/sepformer-libri2mix", savedir="tmpdir",run_opts={"device": "cuda"})
# model = separator.from_hparams(
#     source="speechbrain/sepformer-wham",  # 使用适应多人分离的模型
#     savedir="tmpdir",
#     run_opts={"device": "cuda"}  # 使用GPU进行推理
# )

设备 = "cuda" if torch.cuda.is_available() else "cpu"
# 设备 =  "cpu"
torch数据类型 = torch.float16 if torch.cuda.is_available() else torch.float32
模型ID = "openai/whisper-large-v3-turbo"

# 指定模型下载路径为当前 Python 文件夹
缓存目录 = os.path.dirname(os.path.abspath(__file__))

模型 = AutoModelForSpeechSeq2Seq.from_pretrained(
    模型ID,
    torch_dtype=torch数据类型,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    cache_dir=缓存目录
)
模型.to(设备)

处理器 = AutoProcessor.from_pretrained(模型ID, cache_dir=缓存目录)

流水线 = pipeline(
    "automatic-speech-recognition",
    model=模型,
    tokenizer=处理器.tokenizer,
    feature_extractor=处理器.feature_extractor,
    torch_dtype=torch数据类型,
    device=设备,
    generate_kwargs={"forced_decoder_ids": 处理器.get_decoder_prompt_ids(language="ja", task="transcribe")}
)


# 打开 MP3 文件
file_path = '6657.wav'
target_sample_rate = 16000

# 加载并重采样音频
waveform, sample_rate = torchaudio.load(file_path)
waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)
sample_rate = target_sample_rate

# 确保 waveform 的设备与模型相同
waveform = waveform.mean(dim=0).unsqueeze(0).to("cuda")
print(waveform.shape)
print(f"Model device: {model.device}")
print(f"Waveform device: {waveform.device}")

# 执行分离
output = model(waveform.cuda())
output_tensor = output.squeeze(0).permute(1, 0)
tensor1, tensor2 = torch.unbind(output_tensor, dim=0)

print(float(tensor1.abs().mean().to("cpu")))
print(float(tensor2.abs().mean().to("cpu")))

# 输入1 = tensor1.to("cpu").numpy()
# 输入2 = tensor2.to("cpu").numpy()
#
#
#
# 结果1 = 流水线(输入1)
# print(结果1["text"])
# 结果2 = 流水线(输入2)
# print(结果2["text"])
#
# def 双人流水线(音频,信息 = None):
#     分离模型输入 = torch.tensor(音频).to(设备)
#     分离输出 = model(分离模型输入)
#     分离输出1 = 分离输出.squeeze(0).permute(1, 0)
#     tensor1, tensor2 = torch.unbind(分离输出1, dim=0)
#     输入1 = tensor1.to("cpu").numpy()
#     输入2 = tensor2.to("cpu").numpy()
#     结果1 = 流水线(输入1)
#     结果2 = 流水线(输入2)
#
#     输出 = "说话人1:" + 结果1 + '\n' + "说话人2:" + 结果2
#     return 输出





# 判断输出的轨道数并保存为 wav 文件
num_speakers = output.shape[2]  # 输出张量的第三维表示说话人个数
for i in range(num_speakers):
    # 提取每个说话人的音频轨道
    speaker_audio = output[0, :, i].unsqueeze(0)  # 保持为 [1, num_samples]

    # 保存为 wav 文件
    output_path = f"speaker_{i + 1}_output.wav"
    torchaudio.save(output_path, speaker_audio.cpu(), sample_rate)
    print(f"Saved speaker {i + 1} to {output_path}")

# 打印音频信息
print(f"Waveform shape: {waveform.shape}")  # 打印音频张量的形状
print(f"Sample rate: {sample_rate}")        # 打印采样率
print(f"Output shape: {output.shape}")      # 打印输出张量的形状

