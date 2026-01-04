
# PyTorch Sparse NPU



This package consists of a small extension library of optimized sparse matrix operations with autograd support on Ascend NPU.
This package currently consists of the following methods:

* **[Coalesce](#coalesce)**
* **[Transpose](#transpose)**
* **[Sparse Dense Matrix Multiplication](#sparse-dense-matrix-multiplication)**
* **[Sparse Sparse Matrix Multiplication](#sparse-sparse-matrix-multiplication)**

All included operations work on varying data types and are implemented both for CPU, GPU and Ascend NPU.
To avoid the hazzle of creating [`torch.sparse_coo_tensor`](https://pytorch.org/docs/stable/torch.html?highlight=sparse_coo_tensor#torch.sparse_coo_tensor), this package defines operations on sparse tensors by simply passing `index` and `value` tensors as arguments ([with same shapes as defined in PyTorch](https://pytorch.org/docs/stable/sparse.html)).
Original repo: [torch_sparse](https://github.com/rusty1s/pytorch_sparse)


--------------------------------------------------------------------------------

# torch_sparse NPU 适配文档

## 1. 项目概述

本项目是针对 `torch_sparse` 库的异构计算后端扩展，旨在使其能够高效运行在华为 **Ascend (昇腾) NPU** 硬件平台上。通过对接 **CANN (Compute Architecture for Neural Networks)** 软件栈，实现了稀疏张量算子在 NPU 上的分发与执行。

---

## 2. 核心技术架构

适配工作主要基于 PyTorch 的插件扩展机制，通过以下三个层级的修改实现后端解耦：

### 2.1 算子分发机制 (Dispatching)

在 `csrc/` 核心代码层，将原本的二元分发（CPU/CUDA）升级为三元分发：

* **设备识别**：利用 `tensor.device().type()` 识别 `torch::kPrivateUse1` 标识（该标识由 `torch_npu` 注册给 NPU 设备）。
* **静态路由**：在算子入口文件（如 `convert.cpp`, `spmm.cpp`）中增加 `else if (is_npu)` 逻辑，实现对 NPU 内核函数的路由转发。

### 2.2 构建系统增强 (Build System)

对 `setup.py` 进行了修改，使其具备 NPU 感知能力：

* **自动路径发现**：通过 `import torch_npu` 动态获取 CANN 软件栈在 Python 环境中的安装位置，自动提取 `include` 和 `lib` 路径。
* **NpuExtension 适配**：引入 `torch_npu.utils.cpp_extension.NpuExtension`，相比标准的 `CppExtension`，它能自动注入 NPU 特有的编译宏（如 `WITH_NPU`）和链接库（如 `libascendcl.so`）。

### 2.3 接口实现 (Interface Implementation)

建立了 `csrc/npu/` 目录结构，通过头文件解耦：

* 实现了与 CPU/CUDA 版本完全对称的函数签名。
* 在开发初期建立接口占位符，确保了整体库的可编译性与可链接性，各个算子可单独实现并编译为动态库链接到该库中。

---

## 3. 后续开发计划

目前已完成“脚手架”搭建与分发链路的打通，后续工作将分为三个阶段：

### 第一阶段：算子功能实现

* **ACL 调用**：优先调用 CANN 已有的高阶算子库（如 `aclnn` 接口）实现基础的 `spmm`、`spspmm`。
* **Ascend C 开发**：针对 `ind2ptr`、`ptr2ind` 等细粒度索引变换算子，使用 Ascend C 编写高性能内核。

### 第二阶段：算子性能优化

### 第三阶段：自动化测试与基准测试

* 编写基于 `pytest` 的对齐测试脚本，以 CPU 结果为标杆，验证 NPU 算子的数值精度。
* 进行算子级 Profile 分析，针对 NPU 的异构计算单元优化稀疏矩阵乘法的计算密度。

---

## 4. 编译与安装说明

**前提条件：**

* 已安装 CANN Toolkit。
* 已安装 `torch_npu` 插件。

**执行编译：**

```bash
# 更新子模块
git submodule update --init --recursive

# 开启NPU编译
export FORCE_NPU=1
python setup.py install

```
