# handwriting-ocr

> **当前状态：AI Agent Skill 原型，尚不是可直接安装的 Python 包。**

本仓库目前以 `SKILL.md` 中的工作流说明和代码片段为主体，适合研究手写 OCR、纠错闭环与个性化适配的实现路线。请先阅读以下实现边界，再决定是否把它用于真实数据。

## 实现状态

| 能力 | 当前仓库状态 | 使用建议 |
|---|---|---|
| 基础手写 OCR | 提供基于 PaddleOCR 的示例代码 | 可在本地按所安装的 PaddleOCR 版本验证 |
| `handwriting_ocr` Python API | **未随仓库发布对应模块或包** | `from handwriting_ocr import ...` 不能作为当前仓库的可运行安装示例 |
| 用户纠错记录 | `SKILL.md` 中提供实现代码片段 | 需要自行整理为模块并测试后再用于实际工作 |
| 用户专属微调与推理导出 | 训练和导出代码中的 `subprocess.run` 调用当前被注释 | 不应视为已接通的端到端训练能力 |
| 纠错后持续提升精度 | 设计目标，尚需可执行流程和独立评测证明 | 不应在未经验证时承诺“越用越准” |
| 61 页手写本测试 | `SKILL.md` 记录了作者报告的测试结果 | 仓库未附原始图像、标注、日志和可重跑评测流程，外部读者暂时无法独立复现 |

## 最小 OCR 示例

以下示例直接调用 PaddleOCR，不依赖本仓库不存在的 `handwriting_ocr` 包。它沿用当前 `SKILL.md` 使用的 PaddleOCR v2 风格接口；较新版本的 PaddleOCR 可能需要调整初始化参数。

安装依赖：

```bash
pip install paddleocr paddlepaddle pillow opencv-python-headless pyyaml
```

CPU 示例：

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False)
result = ocr.ocr("handwriting.jpg", cls=True)

for page in result or []:
    for item in page or []:
        box, (text, confidence) = item
        print(f"{text}\t{confidence:.3f}")
```

这是基础 OCR 演示，不代表本仓库已经提供完整的反馈数据库、用户模型隔离、个人微调或模型部署服务。运行前请以已安装版本的 PaddleOCR 文档为准。

## 作为 Agent Skill 使用

如果你的 Agent 支持加载 `SKILL.md`，可以将该文件作为工作流参考。文档中的代码块应被视为实现素材；未被提取为实际模块、执行测试并接入运行流程的功能，不应被视为已交付的 Python API。

## 下一步建议

1. 将经过选择的代码块拆分成真正的 Python 模块，并补上包入口和依赖清单。
2. 将训练、导出命令接入明确的配置、错误处理与超时控制；在端到端测试完成前继续把微调标记为实验性。
3. 提供可合法公开的小型样本、脱敏标注和评测脚本，重新计算 CER 等指标。
4. 对手写图像、纠错文本、用户标识和模型文件建立访问控制、保留期限与删除机制。

## 详细文档

设计说明、示例代码和已知限制见 [SKILL.md](SKILL.md)。
