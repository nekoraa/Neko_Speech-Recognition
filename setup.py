from cx_Freeze import setup, Executable

import sys
sys.setrecursionlimit(200000)

# 设置可执行文件参数
setup(
    name="nekofan",
    version="1.0",
    description="nekofanyi",
    executables=[Executable("语音识别实时翻译GUI.py", base=None, icon="A2E0E5715F34154F12021F72DB63955F.ico")]
)
