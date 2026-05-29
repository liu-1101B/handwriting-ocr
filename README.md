# handwriting-ocr

智能手写识别与个人笔迹适配系统，支持中英文手写识别。

## 功能特性

- ✅ 中英文手写识别（基于PaddleOCR）
- ✅ 用户专属模型微调
- ✅ 纠正反馈闭环（越用越准）
- ✅ Agent集成接口

## 快速开始

```python
from handwriting_ocr import handwriting_ocr, submit_correction

# 识别手写文字
result = handwriting_ocr("handwriting.jpg", user_id="user_123")

# 用户纠正后提交
submit_correction(
    user_id="user_123",
    image_path="handwriting.jpg", 
    ocr_text=result["text"],
    corrected_text="正确文本"
)
```

## 安装

```bash
pip install paddlepaddle paddleocr pyyaml opencv-python-headless pillow
```

## 详细文档

参见 [SKILL.md](SKILL.md)
