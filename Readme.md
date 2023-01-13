# Performance engineering of simple  (Causal interview demo)
THIS CODE EXISTS FOR DEMONSTRATION PURPOSES ONLY AND SHOULD NOT BE USED FOR ANY OTHER PURPOSE, INCLUDING FURTHER DISTRIBUTION.

## Overview

This repository contains a collection of high-performance C++20 implementations of a simple combination of arithmetic operations
on multidimensional (OLAP) cubes of data. Such computation is ubiquitous in financial modeling, where each variable has
multiple dimensions (e.g. cost and revenue can be broken down by regions, products, teams, etc.). This can quickly grow
into millions (or billions) of data points, so performance is of the essence.

In particular, given multidimensional variables `cost[team][geography][time]` and `revenue[team][geography][time][cohort]`,
the code computes `profit = revenue - cost`, SUM-aggregating `revenue` over the cohort dimension.

The code is split across the following files:
- `main.cpp`: handles CLI arguments and contains initialization and benchmarking boilerplate.
- `Tensor.h`: contains the `Tensor<T>` templated class, the abstraction for multidimensional variables.
- `reduceRevenue.h`: contains various implementations of `reduceRevenue`, which performs the computation.
- `test-avx512.cpp`: here is some experimentation code I used to test out the Intel AVX512 intrinsics.

### `Tensor`
The `Tensor<T>` class represents a multidimensional array of type `T`.
It is implemented as a simple wrapper around the `T *data` pointer, which contains the actual data, and provides a 
bunch of utility methods for access and data manipulation. Data can be accessed directly through the raw index
(using `operator[]`) or by specifying each coordinate (using `operator()`). 
The number of dimensions is not a part of the type signature, which is more realistic as we usually wouldn't know it
until runtime. Finally, the class can be properly copied, moved, and printed.

### `reduceRevenue`
This function is templated on the execution policy. That made it easiest to test and compare a bunch of different
implementations, each of which is implemented as a specialization. The function takes in three `Tensor` instances,
`revenue` and `cost` (inputs), and `profit` (output).

The following policies are supported:
- `SERIAL`: a simple nested loops implementation.
- `SERIAL_FAST`: nested loops but with direct raw accesses, which is somehow faster (I think due to vectorization).
- `SERIAL_VEC`: explicit vectorization using AVX-512 intrinsics.
- `PARALLEL`: like `SERIAL_VEC` but the outer loop is parallelized using OpenCilk.
- `PARALLEL_OMP`: I was trying to do an OpenMP benchmark for comparison but couldn't get OpenMP to run on my machine.

I also implemented a helper called `reduce`, which performs a reduction over one row of the tensor and contains
the handwritten vectorization code. It is used in both `SERIAL_VEC` and `PARALLEL` specializations. 

## Performance results
The benchmark was compiled with the OpenCilk 1.1 compiler (based on Clang 12.0.0) with the highest optimization level
(`-O3`) enabled. I ran each benchmark 10 times and took the median running time.
I ran all different implementations on my machine, and also ran the fastest two (`SERIAL_VEC` and `PARALLEL`) on MIT Supercloud.
- My Machine: 11th Generation Intel(R) Core( TM) i9-11900H (24MB Cache, up to 4.9 GHz, 8 cores), 16GB RAM, WSL2, Windows 11
- MIT Supercloud: Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz, 196GB RAM, GridOS 18.04.6

Originally, the problem dimensions were posed with 5 teams, 4 geographies, 2000 time periods, and 2000 cohorts, using
32-bit floats (80M cells total).
`SERIAL` took around 2s on my machine, while `SERIAL_FAST` took 82ms and `SERIAL_VEC` took just 18ms.

For more reliable performance numbers, I increased the problem size to `N=M=20000` (8B cells, 100x original problem size).
On the Supercloud machine, `SERIAL_VEC` took 2497ms. `PARALLEL` took 2530ms on 1 core, and around 400-410ms on more
than 16 cores.

The speedup of `PARALLEL` over `SERIAL_VEC` with respect to the number of cores (up to 48) is shown below.
I believe that the memory bandwidth of the system is saturated around 8-10 cores, which is why the performance plateaus.
There is certainly ample parallelism so the scheduling overheads should be negligible. That's also supported by the
close-to-linear trend in the beginning.

![Median speedup of `PARALLEL` as a function of the number of processors](parallel-speedup.png)

## Build & run
The code is built using CMake. There are CMake targets for each policy:
- `SERIAL`: `causal-serial`
- `SERIAL_FAST`: `causal-serial-fast`
- `SERIAL_VEC`: `causal-serial-vec`
- `PARALLEL`: `causal-cilk`
- `PARALLEL_OMP`: `causal-omp`


```shell
# Configure step
cmake -S . -B cmake-build-release -D CMAKE_BUILD_TYPE=Release

# Build step (for the PARALLEL policy in this case)
cmake --build cmake-build-release --target causal-cilk

# Run:
cmake-build-release/causal-cilk [--help] [--version] [-N VAR] [-M VAR] [-g VAR] [-t VAR]
```

All targets support optionally specifying the sizes of each dimension via the command line.

## Limitations/next steps
I already spent too much time on this, so I had to stop despite wanting to make further improvements.
There are a few more directions in which this could be taken:
1. measure performance with other data types
1. not hard-coding the formula 
1. slightly better data layout
1. sparsity

### 1. Other data types
I believe the largest performance bottlenecks are the memory bandwidth of each core and whole processor,
as well as the width of the vector data lane. 
That means that using 64-bit instead of 32-bit floats would likely double the running time.
Similarly, using `float16` or `bfloat16` would roughly halve the time (at the cost of lower precision).
Using integers might be more beneficial, as integer addition tends to be slightly faster at the same bit-width.
However, those are just guesses and exact performance would need to be measured to be certain.

### 2. Dynamic formulas
By knowing the formula at compile time, it was a lot easier to implement the code and choose the data layout.
In the real world, that isn't the case, so things could get more complicated and potentially there could be overhead.
That could mean that the fastest way for those models would be to just-in-time compile them to machine code.
However, because the memory bandwidth seems to be saturated pretty soon, that might not be as much of an issue.

### 3. Data layout
The performance could potentially be slightly improved by improving the data layout.
Namely, at the end of the loop inside `reduce`, the sum of all elements is contained in a vector register of 16 `float`s
and needs to be reduced down to one, which takes multiple cycles. If instead the code operated on 16 rows at the same time,
that extra reduce operation would not be needed. However, I doubt that would bring a large speedup as it likely takes
a lot less time than the summation of 20k floats in a row.

### 4. Sparsity
Obviously, sparsity could provide enormous speedups, as it could reduce the amount of computation and memory loads and stored.
However, without knowing how sparse the customers' models are, it did not really make any sense to compare that to the dense version.  