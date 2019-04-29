# Compiler Design Project (Main project)

## 项目背景
在课程项目的第一部分中，同学们利用TVM的Tensor描述语言完成了五个卷积神经网络算子前向及反向的实现。其中包含五个算子分别为二维卷积，2x2池化，ReLU，flatten，全连接。通过与PyTorch等框架的对比，这些算子的正确性有了保证。然而，在第一部分中，所实现的算子并没有考虑性能问题。正是因为PyTorch等框架提供了高性能的算子实现，它们才被工业界和学术界广泛应用。为了实现高性能的算子，在课程项目的第二部分，我们需要对算子
进行优化。通过结合两部分课程项目，同学们就有能力搭建完整的卷积神经网络并且完成快速训练与推导。

## 项目要求
本部分是课程项目的第二部分。本部分的目的为设计流程完成对给定算子的性能优化并实现为自动化的程序。

本部分的任务包括：
    1. 学习如何使用TVM来优化算子；
    2. 实现算法完成完成对给定算子的自动优化，有效提升算子性能（即缩短执行时间）

## 使用框架代码
提供了auto\_schedule目录，该目录下有auto\_schedule.py，其中实现了auto\_schedule函数，已经在\_\_init\_\_.py中export，可以在auto\_schedule所在目录调用该包。
如
```python
user@host:$ ls
auto_schedule  config.py  README.md  student_test.py
user@host:$ python
Python 3.5.6 (default, Apr 20 2019, 09:23:07)
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import auto_schedule as auto
>>> auto.auto_schedule
<function auto_schedule at 0x7f57d2d1cc80>
```
同学们实现的auto\_schedule函数及其它辅助结构都要在auto\_schedule目录中定义并使用。

---

**环境**:
Python 3 + Numpy + PyTorch + tvm v0.5

---
**测试**：
```
python student_test.py
```
运行结果会打印出来，并且在project2\_score.txt也记录这这一次的得分。
在config.py中定义了被测试的参数，可以通过在config.py中打注释实现针对某个shape的测试。
测试中是按照随机顺序测试不同参数的，所以请不要对测试顺序有假设。

## 其它
测试框架可能会有考虑不周的地方，有问题请及时反馈。