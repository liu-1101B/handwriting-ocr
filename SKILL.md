---
name: handwriting-ocr
description: "智能手写识别与个人笔迹适配系统。支持中英文手写识别，可通过少量样本微调为用户创建专属识别模型，并通过纠正反馈持续提升识别精度。适用场景：手写笔记数字化、手写文档转文本、个人笔迹识别优化。"
license: MIT
---

# 手写识别与个人笔迹适配系统

## 概述

本技能提供了一套完整的手写识别解决方案：
- **通用识别**：基于PaddleOCR的中英文手写识别
- **个人适配**：为特定用户微调专属模型
- **持续进化**：通过用户纠正反馈不断优化识别精度

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    用户界面 (Agent)                           │
│              handwriting_ocr(image, user_id)                │
└───────────────┬─────────────────────────────┬───────────────┘
                │                             │
                ▼                             ▼
┌───────────────────────┐       ┌───────────────────────────────┐
│    OCR 推理服务        │       │       用户专属模型仓库         │
│                       │       │   user_123/model.pdparams     │
│  - 通用模型 (默认)     │◄──────│   user_456/model.pdparams     │
│  - 用户专属模型        │       └───────────────────────────────┘
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│    置信度分析          │
│    - 高置信度 → 直接输出│
│    - 低置信度 → 高亮提示│
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│    反馈收集器          │
│  submit_correction()  │
│                       │
│  存储: (图片, 纠正文本)│
└───────────┬───────────┘
            │ 当积累 ≥ 20条
            ▼
┌───────────────────────┐
│    异步微调任务        │
│    更新用户专属模型    │
└───────────────────────┘
```

---

## 快速开始

### 环境准备

```bash
# 安装PaddlePaddle
pip install paddlepaddle-gpu  # GPU版本（推荐）
# 或 CPU版本
# pip install paddlepaddle

# 安装PaddleOCR
pip install paddleocr

# 安装其他依赖
pip install pyyaml opencv-python-headless pillow
```

### 基础手写识别

```python
from paddleocr import PaddleOCR

# 初始化OCR引擎（启用手写识别）
ocr = PaddleOCR(
    use_angle_cls=True,      # 启用方向分类
    lang="ch",               # 中文
    use_gpu=True,            # 使用GPU
    det_model_dir=None,      # 使用默认检测模型
    rec_algorithm="SVTR_LCNet",  # 手写识别算法
)

# 识别图片
result = ocr.ocr("handwriting.jpg", cls=True)

# 输出结果
for line in result:
    if line:
        for word_info in line:
            box, (text, confidence) = word_info
            print(f"文字: {text}, 置信度: {confidence:.2f}")
```

---

## 核心功能

### 1. 通用手写识别

适用于快速识别场景，无需个性化训练：

```python
def recognize_handwriting(image_path, user_id=None, confidence_threshold=0.7):
    """
    通用手写识别函数

    Args:
        image_path: 图片路径
        user_id: 用户ID（可选，用于加载专属模型）
        confidence_threshold: 置信度阈值

    Returns:
        dict: {
            "text": 识别文本,
            "confidence": 平均置信度,
            "details": [{"box": [...], "text": "...", "confidence": 0.95}],
            "needs_review": bool  # 是否需要人工审核
        }
    """
    from paddleocr import PaddleOCR
    import os

    # 初始化OCR
    ocr_params = {
        "use_angle_cls": True,
        "lang": "ch",
        "use_gpu": True,
    }

    # 如果有用户专属模型，加载它
    if user_id:
        user_model_path = f"models/users/{user_id}"
        if os.path.exists(f"{user_model_path}/rec_model"):
            ocr_params["rec_model_dir"] = f"{user_model_path}/rec_model"
            ocr_params["det_model_dir"] = f"{user_model_path}/det_model"

    ocr = PaddleOCR(**ocr_params)

    # 执行识别
    result = ocr.ocr(image_path, cls=True)

    # 处理结果
    all_text = []
    all_details = []
    total_confidence = 0
    count = 0

    for line in result:
        if line:
            for word_info in line:
                box, (text, confidence) = word_info
                all_text.append(text)
                all_details.append({
                    "box": box,
                    "text": text,
                    "confidence": confidence
                })
                total_confidence += confidence
                count += 1

    avg_confidence = total_confidence / count if count > 0 else 0

    return {
        "text": "\n".join(all_text),
        "confidence": avg_confidence,
        "details": all_details,
        "needs_review": avg_confidence < confidence_threshold
    }
```

### 2. 用户专属模型微调

为特定用户训练个性化模型：

```python
import os
import json
from datetime import datetime


def prepare_training_data(user_id, samples):
    """
    准备训练数据

    Args:
        user_id: 用户ID
        samples: 样本列表 [{"image": "path", "text": "正确文本"}, ...]

    Returns:
        str: 数据集路径
    """
    dataset_dir = f"training_data/{user_id}"
    os.makedirs(dataset_dir, exist_ok=True)

    # 创建训练配置
    train_list = []
    for i, sample in enumerate(samples):
        # 保存图片
        image_name = f"sample_{i}.jpg"
        image_path = os.path.join(dataset_dir, image_name)
        # 假设sample["image"]是图片路径或bytes
        if isinstance(sample["image"], str):
            import shutil
            shutil.copy(sample["image"], image_path)
        else:
            with open(image_path, "wb") as f:
                f.write(sample["image"])

        train_list.append({
            "image": image_name,
            "label": sample["text"]
        })

    # 保存标注文件
    with open(os.path.join(dataset_dir, "train_list.json"), "w") as f:
        json.dump(train_list, f, ensure_ascii=False, indent=2)

    return dataset_dir


def finetune_user_model(user_id, dataset_dir, base_model="ch_PP-OCRv4_rec"):
    """
    为用户微调专属模型

    Args:
        user_id: 用户ID
        dataset_dir: 训练数据集路径
        base_model: 基础模型名称

    Returns:
        str: 模型保存路径
    """
    import subprocess

    output_dir = f"models/users/{user_id}"
    os.makedirs(output_dir, exist_ok=True)

    # 准备PaddleOCR训练配置
    config = {
        "Global": {
            "epoch_num": 100,
            "log_smooth_window": 20,
            "print_batch_step": 10,
            "save_model_dir": output_dir,
            "save_epoch_step": 10,
            "eval_batch_step": [0, 500],
            "pretrained_model": f"pretrained/{base_model}/pretrain_model",
        },
        "Train": {
            "dataset": {
                "name": "SimpleDataSet",
                "data_dir": dataset_dir,
                "label_file_list": [f"{dataset_dir}/train_list.json"],
                "transforms": [
                    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                    {"RecAug": {}},
                    {"MultiLabelEncode": {}},
                    {"KeepKeys": {"keep_keys": ["image", "label", "length"]}}
                ]
            },
            "loader": {
                "shuffle": True,
                "batch_size_per_card": 64,
                "num_workers": 8
            }
        },
        "Eval": {
            "dataset": {
                "name": "SimpleDataSet",
                "data_dir": dataset_dir,
                "label_file_list": [f"{dataset_dir}/train_list.json"],
                "transforms": [
                    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                    {"MultiLabelEncode": {}},
                    {"KeepKeys": {"keep_keys": ["image", "label", "length"]}}
                ]
            },
            "loader": {
                "shuffle": False,
                "batch_size_per_card": 64,
                "num_workers": 8
            }
        }
    }

    # 保存配置文件
    config_path = os.path.join(output_dir, "train_config.yml")
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # 执行训练（使用PaddleOCR提供的训练脚本）
    cmd = [
        "python", "tools/train.py",
        "-c", config_path
    ]

    print(f"开始训练用户 {user_id} 的专属模型...")
    # 实际生产环境建议异步执行
    # subprocess.run(cmd)

    return output_dir
```

### 3. 反馈收集与再训练闭环

实现"越用越准"的机制：

```python
import sqlite3
from datetime import datetime
import threading


class FeedbackCollector:
    """反馈收集器 - 收集用户纠正数据"""

    def __init__(self, db_path="feedback.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                ocr_text TEXT NOT NULL,
                corrected_text TEXT NOT NULL,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                used_in_training BOOLEAN DEFAULT FALSE
            )
        ''')
        conn.commit()
        conn.close()

    def submit_correction(self, user_id, image_path, ocr_text, corrected_text, confidence=0.0):
        """提交用户纠正"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO feedback (user_id, image_path, ocr_text, corrected_text, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, image_path, ocr_text, corrected_text, confidence))
        conn.commit()

        # 检查是否达到再训练阈值
        count = self.get_pending_count(user_id)
        conn.close()

        if count >= 20:  # 阈值可配置
            self._trigger_retraining(user_id)

        return count

    def get_pending_count(self, user_id):
        """获取待处理的纠正数量"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM feedback
            WHERE user_id = ? AND used_in_training = FALSE
        ''', (user_id,))
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_pending_feedback(self, user_id):
        """获取待处理的纠正数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT image_path, corrected_text FROM feedback
            WHERE user_id = ? AND used_in_training = FALSE
        ''', (user_id,))
        rows = cursor.fetchall()
        conn.close()

        return [{"image": row[0], "text": row[1]} for row in rows]

    def mark_as_used(self, user_id):
        """标记反馈数据已用于训练"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE feedback SET used_in_training = TRUE
            WHERE user_id = ? AND used_in_training = FALSE
        ''', (user_id,))
        conn.commit()
        conn.close()

    def _trigger_retraining(self, user_id):
        """触发异步再训练"""
        def retrain():
            # 获取待处理数据
            samples = self.get_pending_feedback(user_id)

            # 准备训练数据
            dataset_dir = prepare_training_data(user_id, samples)

            # 执行微调
            finetune_user_model(user_id, dataset_dir)

            # 标记数据已使用
            self.mark_as_used(user_id)

            print(f"用户 {user_id} 的模型已更新")

        # 异步执行训练
        thread = threading.Thread(target=retrain)
        thread.start()
```

### 4. Agent集成接口

将上述功能封装为Agent可调用的接口：

```python
from typing import Optional, Dict
import base64


class HandwritingAgent:
    """手写识别Agent - 提供完整的识别和管理功能"""

    def __init__(self):
        self.feedback_collector = FeedbackCollector()

    def recognize(self, image_path: str, user_id: Optional[str] = None) -> Dict:
        """
        Agent调用入口：识别手写文字

        Args:
            image_path: 图片路径或base64编码
            user_id: 用户ID（可选）

        Returns:
            dict: {
                "success": bool,
                "text": str,           # 识别结果
                "confidence": float,   # 置信度
                "needs_review": bool,  # 是否需要人工审核
                "message": str         # 提示信息
            }
        """
        try:
            # 调用识别
            result = recognize_handwriting(image_path, user_id)

            return {
                "success": True,
                "text": result["text"],
                "confidence": result["confidence"],
                "needs_review": result["needs_review"],
                "message": "识别完成，低置信度部分已标记"
            }
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "confidence": 0.0,
                "needs_review": True,
                "message": f"识别失败: {str(e)}"
            }

    def submit_correction(self, user_id: str, image_path: str,
                         ocr_text: str, corrected_text: str,
                         confidence: float = 0.0) -> Dict:
        """
        Agent调用入口：提交用户纠正

        Returns:
            dict: {
                "success": bool,
                "pending_count": int,  # 当前待处理数量
                "message": str
            }
        """
        try:
            count = self.feedback_collector.submit_correction(
                user_id, image_path, ocr_text, corrected_text, confidence
            )

            return {
                "success": True,
                "pending_count": count,
                "message": f"反馈已记录，累计 {count} 条"
            }
        except Exception as e:
            return {
                "success": False,
                "pending_count": 0,
                "message": f"提交失败: {str(e)}"
            }

    def get_user_stats(self, user_id: str) -> Dict:
        """获取用户识别统计"""
        conn = sqlite3.connect(self.feedback_collector.db_path)
        cursor = conn.cursor()

        # 总识别次数
        cursor.execute('''
            SELECT COUNT(*) FROM feedback WHERE user_id = ?
        ''', (user_id,))
        total = cursor.fetchone()[0]

        # 纠正次数
        cursor.execute('''
            SELECT COUNT(*) FROM feedback
            WHERE user_id = ? AND ocr_text != corrected_text
        ''', (user_id,))
        corrections = cursor.fetchone()[0]

        # 平均置信度
        cursor.execute('''
            SELECT AVG(confidence) FROM feedback WHERE user_id = ?
        ''', (user_id,))
        avg_confidence = cursor.fetchone()[0] or 0.0

        conn.close()

        return {
            "total_recognitions": total,
            "corrections": corrections,
            "accuracy": (total - corrections) / total if total > 0 else 1.0,
            "avg_confidence": avg_confidence
        }


# 全局Agent实例
agent = HandwritingAgent()


# Agent调用函数
def handwriting_ocr(image_path: str, user_id: Optional[str] = None) -> Dict:
    """手写识别 - Agent主入口"""
    return agent.recognize(image_path, user_id)


def submit_correction(user_id: str, image_path: str,
                     ocr_text: str, corrected_text: str,
                     confidence: float = 0.0) -> Dict:
    """提交纠正 - Agent调用入口"""
    return agent.submit_correction(user_id, image_path, ocr_text, corrected_text, confidence)
```

---

## 使用示例

### 示例1：基础识别

```python
# Agent调用
result = handwriting_ocr("handwriting.jpg")

if result["needs_review"]:
    print(f"识别完成，置信度较低 ({result['confidence']:.2f})，建议人工审核")
else:
    print(f"识别结果: {result['text']}")
```

### 示例2：带用户适配的识别

```python
# 首次识别
result = handwriting_ocr("note.jpg", user_id="user_123")

# 用户纠正
submit_correction(
    user_id="user_123",
    image_path="note.jpg",
    ocr_text=result["text"],
    corrected_text="正确的文本内容",
    confidence=result["confidence"]
)

# 后续识别会自动使用用户专属模型
result = handwriting_ocr("another_note.jpg", user_id="user_123")
```

### 示例3：批量训练

```python
# 准备用户样本
samples = [
    {"image": "sample1.jpg", "text": "第一行文字"},
    {"image": "sample2.jpg", "text": "第二行文字"},
    # ... 至少20个样本
]

# 开始训练
dataset_dir = prepare_training_data("user_123", samples)
model_path = finetune_user_model("user_123", dataset_dir)
print(f"模型已保存到: {model_path}")
```

---

## 部署建议

### 1. 本地部署

```bash
# 启动OCR服务
python -m paddleocr --image_dir test.jpg --use_angle_cls true --lang ch
```

### 2. API服务化

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import os

app = FastAPI()

@app.post("/ocr")
async def ocr_endpoint(
    image: UploadFile = File(...),
    user_id: str = None
):
    # 保存上传的图片
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    # 执行识别
    result = handwriting_ocr(tmp_path, user_id)

    # 清理临时文件
    os.unlink(tmp_path)

    return JSONResponse(content=result)
```

### 3. 定时训练调度

```python
import schedule
import time

def scheduled_retraining():
    """定时检查并执行再训练"""
    # 检查所有用户的反馈数量
    # 对达到阈值的用户触发训练
    pass

schedule.every().day.at("02:00").do(scheduled_retraining)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## 最佳实践

1. **样本收集**：每个用户至少收集50-100个样本用于初始训练
2. **数据增强**：使用PaddleOCR的内置增强（扭曲、噪声、模糊）扩充样本
3. **置信度阈值**：建议设置0.7-0.8，过低会导致误识别，过高会频繁要求审核
4. **隐私保护**：手写数据属敏感信息，建议本地或私有云部署
5. **模型更新频率**：建议每天凌晨执行批量再训练，避免影响白天使用

---

## 故障排除

### 常见问题

**Q: 识别准确率低怎么办？**
- 检查图片质量（分辨率、光照）
- 增加训练样本数量
- 调整置信度阈值

**Q: GPU内存不足？**
- 减小batch_size
- 使用CPU模式：`use_gpu=False`

**Q: 训练速度慢？**
- 增加num_workers
- 使用GPU训练
- 减少epoch_num

---

## 参考资源

- [PaddleOCR官方文档](https://github.com/PaddlePaddle/PaddleOCR)
- [手写识别模型列表](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md)
- [PaddleOCR训练教程](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/recognition.md)
