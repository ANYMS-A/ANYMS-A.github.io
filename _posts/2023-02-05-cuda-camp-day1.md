---
layout: post
title: "NVIDIA CUDA Camp -- Day 1[新手参加CUDA线上训练营]"
author: "Yalun Hu"
categories: journal
tags: [Blog, CUDA]
image: mountains.jpg
---

## CPU 架构

cpu中较多的晶体管用于数据缓存和流程控制, 只拥有几个少数的高速计算核心. 

- Fetch/Decode: 取指令、译码单元
- ALU(Arithmetic Logic Unit): 算术逻辑单元
- Execution Context: 执行上下文池
- Data cache: 数据缓存
- 流水线优化单元: 如乱序执行、分支断定预测、memory预存取等。


![cpu arch](/assets/img/2023-02-05-cuda-camp-day1/cpu_arch.png)

## 单核（少核）处理器发展的物理约束

$$
P = C * V ^{2} * f
$$

 P 为功耗， V是电压，C是和制程有关的一个常数项，f是时钟频率。

1. 制程极限：量子隧穿。
2. 时钟频率墙：处理器的时钟频率无法保持线性的增长。
3. 存储墙：虽然提高晶体管集成度可以增加处理数据的速度，但是从存储单元读取数据的速度却也是瓶颈。

解决方案之一：
多核心，并行计算。

## GPU 架构

1. 拥有更高的算力
2. 拥有更大的数据传输带宽



![cuda_core_arch](/assets/img/2023-02-05-cuda-camp-day1/cuda_core_arch.png)
- 图中的深浅黄色叠加的小方块，表示的是一个SIMD function unit。（Single Instruction Multiple Data，单指令多数据流，能够复制多个操作数，并把它们打包在大型寄存器的一组指令集。
  ）。Control shared across 16 units(1 MUL-ADD per clock)
  
- Groups of 32[CUDA Threads/fragments/vertices] shared in an instruction stream.

- Up to 48 groups are simultaneously interleaved。

- Up to 1536 individual contexts can be stored。

![cuda_core_arch_detail](/assets/img/2023-02-05-cuda-camp-day1/cuda_core_arch_detail.png)

SM: Stream Multi-Processor

Warp: 32 CUDA Cores。 一个Warp代表了在物理层面，一起同时执行同一个指令的核心们。（虽然逻辑层面我们认为所有thread是并行执行的，但是其实只有一个Warp中的threads在物理层面算是同时执行）
一个warp包含32个并行thread，这32个thread执行于SMIT模式。也就是说所有thread执行同一条指令，
并且每个thread会使用各自的data执行该指令。

**冷知识**：市面上买来的显卡，体积和质量大部分是在风扇和对应的电机，处理器芯片本身的质量是较小的。

## 并行计算简介

- 使用多个计算资源，解决一个计算问题。
- 通信和计算的开销比例要合适。
- 不要受制于访存带宽。

## Amdahl's Law

程序的可能的加速比，取决于可以被并行化的部分：

Speed Up Rate = 1 / (1 - P) 其中， P代表了可以被并行化的部分所占的用时的比例。

如果用N个处理器并行处理：

Speed Up Rate = 1 / (S + (P / N)) 其中，N代表了处理器数量，P是可以并行的部分，S表示串行的部分。


并行化的可扩展行有极限：

|N    |P=0.5|P=0.9|P=0.99|
| --- | --- | --- | ---  |
| 10  | 1.82 | 5.26 | 9.17  |
| 100  | 1.98 | 9.17 | 50.25  |
| 1000  | 1.99 | 9.91 | 90.99  |
| 10000  | 1.99 | 9.91 | 99.02 |


## 初识CUDA

异构计算：简单来说就是将不同的内容分配到不同的设备上进行计算。

- 逻辑控制：CPU。将CPU和内存称作Host。
- 密集计算：GPU。将GPU和显存称作Device。

CUDA安装后如何查看设备中的GPU状态：

- 个人电脑，工作站： nvidia-smi
- Jetson：jtop

CUDA安装后，有很多的sample示例在起安装的文件夹下，可以自行编译，运行，学习。

下图是一个在jetson nano上运行cuda示例程序的一个截图，该程序是用来查看设备各项特性的。

![](/assets/img/2023-02-05-cuda-camp-day1/cuda_example_jetson.png)

下图是一个在作者装有2060显卡上运行同样的示例程序的截图:
![](/assets/img/2023-02-05-cuda-camp-day1/cuda_example_2060.png)

对比之下，是能看出两种设备的资源差异的。



## CUDA编程

编程模式：Extended C

![CUDA Program](/assets/img/2023-02-05-cuda-camp-day1/cuda_program.png)

1. 数据从内存复制到显存。
2. 数据从显存缓存到处理器上，加载GPU程序，执行程序，将结果保存在显存。
3. 将计算结果从显存复制到内存。

何为CUDA的kernel? 以下是摘自CUDA programming guide的一个简介：

"CUDA C++ extends C++ by allowing the programmer to define C++ functions,
 called *kernels*, that, when called, are executed N times in parallel by N different *CUDA threads*,
 as opposed to only once like regular C++ functions.

A kernel is defined using the `__global__` declaration specifier and the number of CUDA threads that
execute that kernel for a given kernel call is specified using a new `<<<...>>>`*execution configuration* syntax
(see [C++ Language Extensions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions)).
Each thread that executes the kernel is given a unique *thread ID* that
is accessible within the kernel through built-in variables "threadIdx"."


Declspecs

```
__device__: 执行空间说明符，功能为:
  - 明确函数需在设备上执行
  - 只能从设备调用
  - __global__和__device__执行空间说明符不可以一起使用
    
__global__: 执行空间说明符，将函数声明为核函数。功能为:
  - 明确函数需在设备上执行
  - 可从主机调用
  - 它修饰的函数必须具有void返回值
  - 对__global__函数的任何调用，都必须指定其执行配置
  - 对__global__函数的调用时异步的，这意味着它在设备完成执行前会返回
  
__host__: 执行空间说明符，功能为:
  - 明确函数需在主机上执行
  - 只能从主机上调用
  - __global__ 和 __host__ 不能一起使用。但是 __device__ 和 __host__ 可以一起使用，在这种情况下，该函数是为主机和设备编译的。
  - 问题: 既然是在主机上执行和调用，和普通的函数有什么区别。这是因为，有些函数，可能既想运行在device端，也想运行在host端，那么如果没有__host__
  说明符，就需要定义两个相同的函数，为在device端运行的加上__device__修饰符，这样做起来有些冗余。因此，就有了__host__就可以让它和__device__说明
  符一起使用，修饰同一个函数，那么编译的时候就会分别为host端和device端都编译一份。
    
__shared__
__local__
__constant__
  ...
```

关键字

```C
threadIdx
blockIdx
```

Intrisics

```c
__syncthreads
```

运行期API

```
Memory: cudaMalloc
symbol

```

## CUDA程序的编译

![](/assets/img/2023-02-05-cuda-camp-day1/cuda_compile_pipe.png)

不同型号的GPU对应了不同类型的架构，因此也对应了不同的编译参数:
关键点：--gpu-architecture参数需要 **小于** --gpu-code

![](/assets/img/2023-02-05-cuda-camp-day1/nvcc_example.png)

关于如何确定自己机器上GPU的architecture, 建议直接搜显卡的型号查看。也有人建议可以使用 cuda device prop查询，但是作者目前为止还没有完全搞明白具体的做法。



nvcc的一些参数，可以通过`nvcc --help` 来查看:

例如，课程示例中涉及到的`--device-c`

  ```
  --device-c                                      (-dc)
          Compile each .c/.cc/.cpp/.cxx/.cu input file into an object file that contains
          relocatable device code.  It is equivalent to '--relocatable-device-code=true
          --compile'
  ```

**疑惑点**： 课程中提及的 真实架构 和 虚拟架构 的概念没有搞明白。

## NVPROF

nvprof 是NVIDIA提供的用于生成GPU timeline的工具，它是cuda toolkit自带的。

Kernel timeline输出的是以GPU kernel为单位的一段时间的运行时间线。我们可以通过其观察GPU在什么时候闲置或者利用不充分，更准确的定位优化的问题。

使用时:

`nvprof ./executable`

关于nvprof的参数，同样也可以通过 `nvprof --help进行查看`

