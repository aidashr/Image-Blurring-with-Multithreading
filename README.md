# Multithreaded Image Blurring Using C++ and OpenCV

This project implements a **manual image blurring (convolution) algorithm** in C++ using a fixed 5×5 kernel and **CPU multithreading** with `std::thread`. The goal is to demonstrate how image processing workloads can be parallelized at the pixel level to improve performance on multi-core systems.

The implementation avoids built-in OpenCV blur functions and instead applies convolution explicitly to highlight low-level computation and parallel execution behavior.

---

## Project Overview

- Language: **C++**
- Libraries: **:contentReference[oaicite:0]{index=0}**
- Parallelism: `std::thread`
- Build System: **CMake**
- Input: RGB images
- Output: Blurred images + execution time measurements

---

## Algorithm Description

### Image Blurring (Convolution)
The program applies a **5×5 averaging kernel**:

- Each output pixel is computed as the weighted sum of its neighboring pixels
- Boundary pixels are excluded to avoid invalid memory access
- The operation is applied independently to each color channel (RGB)

### Parallelization Strategy

- The image is divided **row-wise**
- Each thread processes a disjoint range of rows
- Threads execute the convolution independently
- Results are written into a shared output image without race conditions

This approach exploits **data parallelism**, as each pixel computation is independent.

---

## Implementation Details

- The kernel is statically defined as a normalized 5×5 matrix
- Timing is measured using `std::chrono`
- Thread count can be adjusted in the code to evaluate performance scalability
- The program uses OpenCV only for:
  - Image I/O
  - Matrix representation (`cv::Mat`)

---

## Build Requirements

- C++17 compatible compiler
- OpenCV installed and properly linked
- CMake ≥ 3.10

---

## Build and Run Instructions

### 1. Configure the Project

```bash
mkdir build
cd build
cmake ..
```
### 2. Compile

```bash
make
```
### 3. Run

```bash
./blur


