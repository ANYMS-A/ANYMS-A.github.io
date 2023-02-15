---
layout: post
title: "NVIDIA CUDA Camp -- Day 2[新手参加CUDA线上训练营]"
author: "Yalun Hu"
categories: journal
tags: [Blog, CUDA]
image: mountains.jpg
--- 

## GPU内存的种类以及每个线程对它的读写权限

1. Register
2. Shared Memory
3. Local Memory
4. Global Memory
5. Constant Memory
6. Texture Memory


![cpu arch](/assets/img/2023-02-07-cuda-camp-day2/memories_type.png)
<div style="text-align: center;">课件截图</div>


## 内存管理API
```
// CPU
malloc()
memset()
free()

//GPU
cudaMalloc()
cudaMemset()
cudaFree()

cudaHostAlloc() // for allocating Pinned(Page-locked) memory.

```

## 设备间的数据传输

使用`cudaMemcpy`来进行设备间的数据拷贝

![cpu arch](/assets/img/2023-02-07-cuda-camp-day2/memCpy.png)
<div style="text-align: center;">课件截图</div>


## 矩阵乘法的例子

对于矩阵乘法，如果使用单核CPU处理器，单线程计算一个 A @ B = C的矩阵乘法

假设 A 为 M * N, B 为 N * K的维度: 

CPU计算所需的时间复杂度为O(M * N * K)

而如果使用GPU进行计算，那么我们可以：
- 每个线程，负责计算C中的一个元素。
- 每个线程，读取A中的一行数据，再读取B中的一列数据。
- 为每一行（i），每一列(j)，中对应的元素执行乘法，并加到一个变量中，作为C[i][j]的值

C矩阵中，对应的行列坐标和线程id的关系如下:

cuda线程排布中, x为水平方向, y为垂直方向 则: 


```c++
    Thread_x(row) = blockIdx.y * blockDim.y + threadIdx.y; 
    Thread_y(col) = blockIdx.x * blockDim.x + threadIdx.x;
```

![cpu arch](/assets/img/2023-02-07-cuda-camp-day2/matrix_index_calculation.png)
<div style="text-align: center;">课件截图</div>


## 矩阵乘法代码示例

```c++

__global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

```

## CUDA运行时错误检测

![cpu arch](/assets/img/2023-02-07-cuda-camp-day2/cuda_errors.png)
<div style="text-align: center;">课件截图</div>

课程中提供了一个名为`error.cuh`的头文件，定义了一个可以检测CUDA运行时错误的函数，能够帮助我们发现错误。

```c++
#pragma once
#include <stdio.h>

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)
```

## CUDA Event

CUDA Event的本质是一个GPU时间戳，这个时间戳是在用户指定的时间点上记录的。
因CPU和GPU之间是异步的，统计GPU程序的运行时间最好是由GPU上的时钟来作为参考。
由于GPU本身支持时间戳的记录，因此避免了使用CPU定时器来统计GPU执行时间可能遇到的诸多问题。

![cpu arch](/assets/img/2023-02-07-cuda-camp-day2/cuda_events.png)
<div style="text-align: center;">课件截图</div>

代码示例，利用CUDA Event来监测核函数的执行时间

```c++
cudaEvent_t start, stop;
CHECK(cudaEventCreate(&start));
CHECK(cudaEventCreate(&stop));

CHECK(cudaEventRecord(start));
//cudaEventQuery(start);
gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);    
CHECK(cudaEventRecord(stop));
CHECK(cudaEventSynchronize(stop));
float elapsed_time;
CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

```

- cudaEventRecord(start):
  将start放到 默认stream中，因为我们没创建stream，所以是在默认stream。
  当这个start到stream的时候，就会在device上记录一个时间戳。
  cudaEventRecord(）视为一条记录当前时间的语句，并且把这条事件放入GPU的未完成队列中。
  因为直到GPU执行完了在调用cudaEventRecord(）之前的所有语句时，事件才会被记录下来。
  
- cudaEventRecord(stop)
  记录到的stop，是device执行完之后才会将事件加入到device。 
  所以cudaEventElapsedTime记录的事件start，和stop的时间就是device在某个stream的执行时间。
  
- cudaEventSynchronize(stop)
  会阻塞CPU，直到特定的event被记录。
  也就是这里会阻塞，直到stop在stream中被记录才会向下执行。
  不使用这句话的话，kernel是异步的，还没执行完，CPU就继续往下走了。
  那么cudaEventElapsedTime就记录不到时间了。
  因为有可能stop事件还没加入到device中。



