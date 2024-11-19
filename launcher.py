import subprocess
import sys
import os

# 获取当前目录
current_directory = os.path.dirname(os.path.abspath(__file__))

# 执行 python 语音识别实时翻译GUI.py 命令
subprocess.run([sys.executable, os.path.join(current_directory, "语音识别实时翻译多人测试GUI.py")])
