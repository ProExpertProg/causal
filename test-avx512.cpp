#include <iostream>
#include <random>
#include <chrono>
#include <cassert>
#include <immintrin.h>

constexpr long LENGTH = 2000000;
double array[16 * LENGTH], out[8 * LENGTH]; // make sure they're not removed

int main(int argc, char *argv[]) {
    if (argc <= 1) return -1;
    double factor = std::stod(argv[1]);

    for (int i = 0; i < 16 * LENGTH; ++i) {
        array[i] = factor * i;
    }

    std::chrono::nanoseconds duration[10];

    for (int j = 0; j < 10; ++j) {
        auto start = std::chrono::high_resolution_clock::now();
#ifdef COMPILER
        for (int i = 0; i < 8 * LENGTH; ++i) {
            out[i] = array[i] + array[8 * LENGTH + i];
        }
#else
        for (int i = 0; i < LENGTH; ++i) {
            __m512d arr = _mm512_loadu_pd(&array[8 * i]);
            __m512d arr2 = _mm512_loadu_pd(&array[8 * (LENGTH + i)]);

            __m512d result = _mm512_add_pd(arr, arr2);
            _mm512_storeu_pd(&out[8 * i], result);
        }
#endif
        auto end = std::chrono::high_resolution_clock::now();
        duration[j] = end - start;
    }

    for (auto d: duration) {
        std::cout << duration_cast<std::chrono::duration<float, std::milli>>(d).count() << "ms" << std::endl;
    }
//    for (auto &item: out) {
//        std::cout << item << " ";
//    }
    std::cout << std::endl;
    return 0;
}
