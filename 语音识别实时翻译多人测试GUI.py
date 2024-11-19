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
from speechbrain.inference import SepformerSeparation as separator

model = separator.from_hparams(source="speechbrain/sepformer-libri2mix", savedir="tmpdir", run_opts={"device": "cuda"})
阈值长度 = 10
实时翻译速率 = 4
识别人数 = 1


def 双人流水线(音频, 信息=None):
    分离模型输入 = torch.tensor(音频).unsqueeze(0).to("cuda")
    # print(分离模型输入.shape)
    分离输出 = model(分离模型输入)
    分离输出1 = 分离输出.squeeze(0).permute(1, 0)
    tensor1, tensor2 = torch.unbind(分离输出1, dim=0)
    输出1平均 = tensor1.abs().mean().to("cpu")
    输出2平均 = tensor2.abs().mean().to("cpu")

    if float(输出1平均) > 0.2:
        输入1 = tensor1.to("cpu").numpy()
        结果1 = 流水线(输入1, generate_kwargs={"language": 信息})
    else:
        结果1 = {"text": "无声音"}
    if float(输出2平均) > 0.2:
        输入2 = tensor2.to("cpu").numpy()
        结果2 = 流水线(输入2, generate_kwargs={"language": 信息})
    else:
        结果2 = {"text": "无声音"}

    输出 = "说话人1:" + 结果1["text"] + '\n' + "说话人2:" + 结果2["text"]
    return 输出


停止参数 = 0
语言 = "ja"
translator = Translator()
打印字符 = ""
打印字符1 = ""
# 设置设备和模型
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
    generate_kwargs={"forced_decoder_ids": 处理器.get_decoder_prompt_ids()}
    # generate_kwargs={"forced_decoder_ids": 处理器.get_decoder_prompt_ids(language="ja", task="transcribe")}
)


def 修剪字符串(输入字符串, 最大长度):
    """
    当字符串超过指定长度时，只保留后面的内容。
    :param 输入字符串: 原始字符串
    :param 最大长度: 最大长度
    :return: 修剪后的字符串
    """
    if len(输入字符串) > 最大长度:
        return 输入字符串[-最大长度:]
    return 输入字符串


def 添加文本(新文本):
    文本框.clear()

    """
    追加新内容到文本框
    """
    当前文本 = 文本框.toPlainText()

    if 当前文本:
        更新后的文本 = 当前文本 + "\n" + 新文本
    else:
        更新后的文本 = 新文本
    文本框.setPlainText(更新后的文本)


def 添加文本2(新文本):
    文本框2.clear()
    """
    追加新内容到文本框
    """
    当前文本 = 文本框2.toPlainText()

    if 当前文本:
        更新后的文本 = 当前文本 + "\n" + 新文本
    else:
        更新后的文本 = 新文本
    文本框2.setPlainText(更新后的文本)


def 添加文本3(新文本):
    文本框3.clear()
    """
    追加新内容到文本框
    """
    当前文本 = 文本框3.toPlainText()

    if 当前文本:
        更新后的文本 = 当前文本 + "\n" + 新文本
    else:
        更新后的文本 = 新文本
    文本框3.setPlainText(更新后的文本)


def 识别模型(x1):
    global 打印字符1

    if 识别人数 == 1:
        with torch.no_grad():
            # 输入数据 = 输入数据.to(设备)  # 将输入传送到GPU
            结果 = 流水线(x1, generate_kwargs={"language": 语言})  # 在GPU上进行处理

        # 打印识别出的文本
        sys.stdout.write(Fore.WHITE + f'\r' + 结果["text"])
        打印字符1 = ""
        打印字符1 = 结果["text"]
        # print("\n识别结果：", 结果["text"])
        # return 流水线(x1)
    elif 识别人数 == 2:
        with torch.no_grad():
            # 输入数据 = 输入数据.to(设备)  # 将输入传送到GPU
            结果 = 双人流水线(x1, 语言)  # 在GPU上进行处理

        # 打印识别出的文本
        sys.stdout.write(Fore.WHITE + f'\r' + 结果)
        打印字符1 = ""
        打印字符1 = 结果
        # print("\n识别结果：", 结果["text"])
        # return 流水线(x1)
        return 0


def 识别模型翻译(x1, 启用=0):
    global 打印字符
    global 打印字符1
    if 识别人数 == 1:
        with torch.no_grad():
            # 输入数据 = 输入数据.to(设备)  # 将输入传送到GPU
            结果 = 流水线(x1, generate_kwargs={"language": 语言})  # 在GPU上进行处理

        # 打印识别出的文本
        sys.stdout.write(f'\r ')
        print(Fore.BLUE + 结果["text"])
        打印字符 = 结果["text"] + "\n" + 打印字符
        打印字符1 = ""
        打印字符1 = 结果["text"]
        # 添加文本(结果["text"])
        if 启用 == 1:
            translation = translator.translate(结果["text"], src='ja', dest='zh-cn')
            print(Fore.YELLOW + "翻译:" + translation.text)

    elif 识别人数 == 2:

        with torch.no_grad():
            # 输入数据 = 输入数据.to(设备)  # 将输入传送到GPU
            结果 = 双人流水线(x1, 语言)  # 在GPU上进行处理

        # 打印识别出的文本
        sys.stdout.write(f'\r ')
        print(Fore.BLUE + 结果)
        打印字符 = 结果 + "\n" + 打印字符
        # 添加文本(结果["text"])
        if 启用 == 1:
            translation = translator.translate(结果, src='ja', dest='zh-cn')
            print(Fore.YELLOW + "翻译:" + translation.text)

        return 0


# 音频录制参数
采样率 = 16000  # Whisper 模型要求的采样率为 16kHz
时长 = 0.5  # 每次录制 0.2 秒音频块（用于检测）

# 声音活动检测的阈值，值越大越敏感
声音阈值 = 0.0086
静音阈值 = 5  # 检测到多少次连续静音后停止录音


def 是否有声音活动(音频):
    """通过计算音频信号能量来判断是否有声音活动"""
    return np.mean(np.abs(音频)) > 声音阈值


def 列出音频设备():
    """列出可用的音频输入设备"""
    设备列表 = sd.query_devices()
    输入设备 = [设备 for 设备 in 设备列表 if 设备['max_input_channels'] > 0]

    print("可用的音频输入设备:")
    for i, 设备 in enumerate(输入设备):
        print(f"{i + 1}: {设备['name']} (ID: {设备['index']})")

    return 输入设备


def 选择音频设备(设备列表):
    """让用户选择音频设备"""
    while True:
        try:
            选择 = int(input("请选择音频设备 (输入设备编号): ")) - 1
            if 0 <= 选择 < len(设备列表):
                return 设备列表[选择]['index']  # 返回设备的索引
            else:
                print("无效的设备编号，请重试。")
        except ValueError:
            print("请输入一个有效的数字。")


def 录音并识别(输入设备):
    """录音并进行语音识别"""
    音频缓冲区 = []
    静音计数 = 0
    正在录音 = False
    global 实时翻译速率
    global 阈值长度
    global 计数次数

    计数次数 = 0
    while True:

        if 识别人数 == 1:
            实时翻译速率 = 2
            阈值长度 = 25
        elif 识别人数 == 2:
            实时翻译速率 = 4
            阈值长度 = 10

        总时长 = sum(len(块) for 块 in 音频缓冲区) / 采样率
        while 总时长 > 阈值长度:
            # 移除最旧的一块音频
            音频缓冲区.pop(0)
            总时长 = sum(len(块) for 块 in 音频缓冲区) / 采样率

        # 录制一小块音频
        音频 = sd.rec(int(时长 * 采样率), samplerate=采样率, channels=1, dtype="float32", device=输入设备)
        sd.wait()

        音频 = np.squeeze(音频)  # 去除多余的维度

        # 检测是否有声音活动
        if 是否有声音活动(音频):

            音频缓冲区.append(音频)  # 将有声音的音频片段加入缓冲区



            正在录音 = True
            静音计数 = 0

            if 计数次数 % 实时翻译速率 == 0:
                流处理1 = np.concatenate(音频缓冲区)
                threading.Thread(target=识别模型, args=(流处理1,)).start()

            计数次数 = 计数次数 + 1

            # print("检测到声音，正在录音...")
        elif 正在录音:

            音频缓冲区.append(音频)

            静音计数 += 1
            # print(f"静音检测中...{静音计数}")

            if 计数次数 % 实时翻译速率 == 0:
                流处理2 = np.concatenate(音频缓冲区)
                threading.Thread(target=识别模型, args=(流处理2,)).start()

            计数次数 = 计数次数 + 1

            if 静音计数 >= 静音阈值:
                # 认为语音结束，开始识别
                计数次数 = 0
                break
        else:
            pass

    if 音频缓冲区:
        # 将缓冲区中的音频片段拼接在一起
        全部音频 = np.concatenate(音频缓冲区)
        sys.stdout.write(f'\r ')
        print(type(全部音频))
        print(全部音频.shape)
        threading.Thread(target=识别模型翻译, args=(全部音频,)).start()
        音频缓冲区 = []


# if __name__ == "__main__":
#     # 列出音频设备并选择
#     设备列表 = 列出音频设备()
#     选择的设备 = 选择音频设备(设备列表)
#
#     while True:
#         录音并识别(选择的设备)

def 开始(选择的设备):
    global 停止参数
    while True:
        if 停止参数 == 0:
            录音并识别(选择的设备)
        else:
            停止参数 = 0
            break


def 更新标签():
    global 静音阈值
    """
    更新标签的文本，显示滑块的当前值
    """
    静音阈值 = int(滑块.value())
    标签1.setText(f"设置检测时间长度: {滑块.value()}")

def 更新人数():
    global 识别人数
    """
    更新标签的文本，显示滑块的当前值
    """
    识别人数 = int(人数滑块.value())
    人数选择.setText(f"人数选择: {人数滑块.value()}")


def 更新标签2():
    global 声音阈值
    """
    更新标签的文本，显示滑块的当前值
    """
    声音阈值 = float(滑块2.value() / 10000)
    标签2.setText(f"设置声音阈值: {滑块2.value()}")


def 更新标签3():
    """
    更新标签的文本，显示滑块的当前值
    """

    标签4.setText(f"窗口透明度: {滑块3.value()}")
    窗口2.setWindowOpacity(float(滑块3.value()) / 100)


def 更新语言():
    global 语言
    选中的项 = 语言下拉框.currentText()
    语言 = 选中的项


def 停止选项():
    global 停止参数
    停止参数 = 1
    提交按钮.setEnabled(True)
    提交按钮.setStyleSheet("""
                QPushButton {
                    background-color: white;   /* 按钮背景色 */
                }
                QPushButton:hover {
                    background-color: lightgrey; /* 鼠标悬停颜色 */
                }
                QPushButton:pressed {
                    background-color: darkgrey;  /* 按下时的颜色 */
                }
            """)


def 提交选项():
    提交按钮.setEnabled(False)
    提交按钮.setStyleSheet("""
                QPushButton {
                    background-color: darkgrey;   /* 按钮背景色 */
                }
                QPushButton:hover {
                    background-color: lightgrey; /* 鼠标悬停颜色 */
                }
                QPushButton:pressed {
                    background-color: darkgrey;  /* 按下时的颜色 */
                }
            """)
    """
    获取下拉框选中的项，并作为参数传入函数
    """
    选中的项 = 下拉框.currentText()  # 获取下拉框当前选中的文本
    匹配结果 = re.match(r"(\d+):", 选中的项)
    global 语言
    选中的项 = 语言下拉框.currentText()
    语言 = 选中的项
    数字 = 匹配结果.group(1)

    选择 = int(数字) - 1
    if 0 <= 选择 < len(设备列表):
        print(f"你选择的选项是：{设备列表[选择]['index']}")
        threading.Thread(target=开始, args=(设备列表[选择]['index'],)).start()
        # 开始(设备列表[选择]['index'])

    else:
        print("无效的设备编号，请重试。")


def update_ui(result):
    pass


# 创建应用程序
应用程序 = QApplication(sys.argv)

# 创建主窗口部件
窗口 = QWidget()
窗口.setWindowTitle("NEKO实时翻译系统")
窗口.resize(400, 500)
窗口.setWindowIcon(QIcon('A2E0E5715F34154F12021F72DB63955F.jpg'))
# 窗口.setStyleSheet("""
#     QWidget {
#         background-image: url('img.png');
#         background-repeat: no-repeat;
#         background-size: cover;
#         background-position : center;
#     }
#
#     QLabel, QPushButton {
#         background: transparent;  /* 确保子组件背景不受影响 */
#     }
# """)

窗口2 = QWidget()
窗口2.setWindowTitle("PyQt6 简化版示例")
窗口2.resize(1000, 100)
窗口2.setWindowOpacity(0.8)
窗口2.setWindowFlags(窗口2.windowFlags() | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)

鼠标初始位置 = QPoint()


def 鼠标按下事件(event):
    global 鼠标初始位置
    if event.button() == Qt.MouseButton.LeftButton:
        鼠标初始位置 = event.globalPosition().toPoint() - 窗口2.pos()
        event.accept()


def 鼠标移动事件(event):
    if event.buttons() == Qt.MouseButton.LeftButton:
        窗口2.move(event.globalPosition().toPoint() - 鼠标初始位置)
        event.accept()


# 绑定鼠标事件
窗口2.mousePressEvent = 鼠标按下事件
窗口2.mouseMoveEvent = 鼠标移动事件

# 创建布局
布局 = QVBoxLayout()
布局2 = QVBoxLayout()

# 创建一个只读的多行文本框
文本框 = QTextEdit()
文本框.setReadOnly(True)

文本框.setStyleSheet("""
            QTextEdit {
                background-color: black;  /* 背景颜色 */
                color: white;            /* 字体颜色 */
                font-weight: bold;       /* 加粗字体 */
                font-size: 18px;         /* 字体大小 */
            }
        """)

文本框3 = QTextEdit()
文本框3.setReadOnly(True)

文本框3.setStyleSheet("""
            QTextEdit {
                background-color: black;  /* 背景颜色 */
                color: white;            /* 字体颜色 */
                font-weight: bold;       /* 加粗字体 */
                font-size: 20px;         /* 字体大小 */
            }
        """)
布局2.addWidget(文本框3)
布局.addWidget(文本框)

文本框2 = QTextEdit()
文本框2.setReadOnly(True)

文本框2.setStyleSheet("background: transparent;")

布局.addWidget(文本框2)

水平布局 = QHBoxLayout()

# 创建一个标签，用作下拉框的名称
标签 = QLabel("选择设备:")
水平布局.addWidget(标签)

# 创建下拉框并添加选项
下拉框 = QComboBox()
下拉框.setStyleSheet("background: transparent;")
设备列表 = 列出音频设备()

for i, 设备 in enumerate(设备列表):
    下拉框.addItem(f"{i + 1}: {设备['name']} (ID: {设备['index']})")

水平布局.addWidget(下拉框)
水平布局.addStretch()
# 将水平布局添加到垂直布局中
布局.addLayout(水平布局)

语言水平布局 = QHBoxLayout()

# 创建一个标签，用作下拉框的名称
语言标签 = QLabel("选择语言:")
语言水平布局.addWidget(语言标签)
人数选择 = QLabel("人数选择:1")


# 创建下拉框并添加选项
语言下拉框 = QComboBox()
语言下拉框.setStyleSheet("background: transparent;")

语言下拉框.addItem("english")
语言下拉框.addItem("chinese")
语言下拉框.addItem("japanese")
语言下拉框.addItem("korean")
语言下拉框.addItem("russian")

语言水平布局.addWidget(语言下拉框)
语言水平布局.addWidget(人数选择)


人数滑块 = QSlider(Qt.Orientation.Horizontal)  # 水平滑块
人数滑块.setMinimum(1)  # 最小值
人数滑块.setMaximum(2)  # 最大值
人数滑块.setSingleStep(1)  # 每次滑动的步进值
人数滑块.setValue(1)  # 初始值
语言水平布局.addWidget(人数滑块)
人数滑块.valueChanged.connect(更新人数)


语言水平布局.addStretch()
# 将水平布局添加到垂直布局中
布局.addLayout(语言水平布局)

水平布局2 = QHBoxLayout()

标签1 = QLabel("设置检测时间长度:5")
水平布局2.addWidget(标签1)

滑块 = QSlider(Qt.Orientation.Horizontal)  # 水平滑块
滑块.setMinimum(1)  # 最小值
滑块.setMaximum(20)  # 最大值
滑块.setSingleStep(1)  # 每次滑动的步进值
滑块.setValue(5)  # 初始值
水平布局2.addWidget(滑块)
布局.addLayout(水平布局2)

滑块.valueChanged.connect(更新标签)
滑块.setStyleSheet("background: transparent;")
水平布局3 = QHBoxLayout()
标签2 = QLabel("设置声音阈值:86")
水平布局3.addWidget(标签2)

滑块2 = QSlider(Qt.Orientation.Horizontal)  # 水平滑块

滑块2.setMinimum(1)  # 最小值
滑块2.setMaximum(200)  # 最大值
滑块2.setSingleStep(1)  # 每次滑动的步进值
滑块2.setValue(86)  # 初始值
水平布局3.addWidget(滑块2)
布局.addLayout(水平布局3)
滑块2.setStyleSheet("background: transparent;")
滑块2.valueChanged.connect(更新标签2)

提交按钮 = QPushButton("开始")

提交按钮.setStyleSheet("""
            QPushButton {
                background-color: white;   /* 按钮背景色 */
            }
            QPushButton:hover {
                background-color: lightgrey; /* 鼠标悬停颜色 */
            }
            QPushButton:pressed {
                background-color: darkgrey;  /* 按下时的颜色 */
            }
        """)

提交按钮.clicked.connect(提交选项)
布局.addWidget(提交按钮)

停止按钮 = QPushButton("停止")

停止按钮.setStyleSheet("""
            QPushButton {
                background-color: white;   /* 按钮背景色 */
            }
            QPushButton:hover {
                background-color: lightgrey; /* 鼠标悬停颜色 */
            }
            QPushButton:pressed {
                background-color: darkgrey;  /* 按下时的颜色 */
            }
        """)

停止按钮.clicked.connect(停止选项)
布局.addWidget(停止按钮)

水平布局字幕设置 = QHBoxLayout()
标签3 = QLabel("字幕开关")
水平布局字幕设置.addWidget(标签3)
开关按钮 = QToolButton()
开关按钮.setText("关")
开关按钮.setCheckable(True)
开关按钮.setChecked(True)  # 默认开启
开关按钮.setStyleSheet("background: transparent;")


# 设置按钮样式

# 按钮点击切换文本
def 切换文本():
    if 开关按钮.isChecked():
        开关按钮.setText("关")
        窗口2.hide()
    else:
        开关按钮.setText("开")
        窗口2.show()


开关按钮.clicked.connect(切换文本)

# 创建布局并添加按钮

水平布局字幕设置.addWidget(开关按钮)

布局.addLayout(水平布局字幕设置)

水平布局4 = QHBoxLayout()

标签4 = QLabel("窗口透明度:10")
水平布局4.addWidget(标签4)

滑块3 = QSlider(Qt.Orientation.Horizontal)  # 水平滑块
滑块3.setMinimum(1)  # 最小值
滑块3.setMaximum(100)  # 最大值
滑块3.setSingleStep(1)  # 每次滑动的步进值
滑块3.setValue(80)  # 初始值
水平布局4.addWidget(滑块3)
布局.addLayout(水平布局4)

滑块3.valueChanged.connect(更新标签3)
滑块3.setStyleSheet("background: transparent;")
# 设置窗口的布局


定时器 = QTimer()
定时器.timeout.connect(lambda: 添加文本(修剪字符串(打印字符1, 100)))
定时器.start(100)  # 每1000毫秒（即1秒）执行一次

定时器1 = QTimer()
定时器1.timeout.connect(lambda: 添加文本2(打印字符))
定时器1.start(1000)  # 每1000毫秒（即1秒）执行一次

定时器2 = QTimer()
定时器2.timeout.connect(lambda: 添加文本3(修剪字符串(打印字符1, 100)))
定时器2.start(50)  # 每1000毫秒（即1秒）执行一次

# 设置布局到窗口并显示
窗口.setLayout(布局)
窗口.show()
窗口2.setLayout(布局2)
窗口2.hide()

# 运行应用程序
sys.exit(应用程序.exec())
