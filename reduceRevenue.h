#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
#pragma once

#include <cilk/cilk.h>
#include <immintrin.h>

enum class Policy {
    SERIAL,
    SERIAL_FAST,
    SERIAL_VEC,
    PARALLEL,
    PARALLEL_OMP,
    LENGTH
};


template<Policy p>
__always_inline void reduceRevenue(const Tensor<float> &revenue, const Tensor<float> &cost, Tensor<float> &profit,
                                   size_t N, size_t M, size_t geos, size_t teams);

template<>
__always_inline void
reduceRevenue<Policy::SERIAL>(const Tensor<float> &revenue, const Tensor<float> &cost, Tensor<float> &profit,
                              size_t N, size_t M, size_t geos, size_t teams) {
    for (size_t geo = 0; geo < geos; ++geo) {
        for (size_t team = 0; team < teams; ++team) {
            for (size_t time = 0; time < N; ++time) {
                for (size_t cohort = 0; cohort < M; ++cohort) {
                    profit(geo, team, time) += revenue(geo, team, time, cohort);
                }
                profit(geo, team, time) -= cost(geo, team, time);
            }
        }
    }
}

template<>
__always_inline void
reduceRevenue<Policy::SERIAL_FAST>(const Tensor<float> &revenue, const Tensor<float> &cost, Tensor<float> &profit,
                                   size_t N, size_t M, size_t geos, size_t teams) {
    for (size_t i = 0; i < geos * teams * N; ++i) {
        for (size_t cohort = 0; cohort < M; ++cohort) {
            profit[i] += revenue[i * M + cohort];
        }
        profit[i] -= cost[i];
    }
}

template<>
__always_inline void
reduceRevenue<Policy::SERIAL_VEC>(const Tensor<float> &revenue, const Tensor<float> &cost, Tensor<float> &profit,
                                  size_t N, size_t M, size_t geos, size_t teams) {
    for (size_t i = 0; i < geos * teams * N; ++i) {
        // not proud of these magic constants here
        constexpr size_t N_PACKED_FLOATS = 16, // 512/32
        MASK_TAIL = 15ul, // remainder after the loop
        MASK_VEC = ~MASK_TAIL;

        __m512 sum = _mm512_set1_ps(0.0f);

        for (size_t cohort = 0; cohort < (M & MASK_VEC); cohort += N_PACKED_FLOATS) {
            // profit[i] += revenue[i * M + cohort];
            // but do N_PACKED_FLOATS iterations in parallel
            __m512 rev = _mm512_loadu_ps(&revenue[i * M + cohort]);
            sum = _mm512_add_ps(sum, rev);
        }

        // tail, add remaining elements
        size_t remainder = M % N_PACKED_FLOATS;
        if (remainder) {
            uint16_t m = 0xFFFFu >> (N_PACKED_FLOATS - remainder);
            auto mask = _load_mask16(&m);
            __m512 rev = _mm512_loadu_ps(&revenue[i * M + (M & MASK_VEC)]);
            sum = _mm512_mask_add_ps(sum, mask, sum, rev);
        }

        // sum is in vector form,
        profit[i] += _mm512_reduce_add_ps(sum) - cost[i];
    }
}

template<>
__always_inline void
reduceRevenue<Policy::PARALLEL>(const Tensor<float> &revenue, const Tensor<float> &cost, Tensor<float> &profit,
                                size_t N, size_t M, size_t geos, size_t teams) {
    cilk_for (size_t i = 0; i < geos * teams * N; ++i) {
        // not proud of these magic constants here
        constexpr size_t N_PACKED_FLOATS = 16, // 512/32
        MASK_TAIL = 15ul, // remainder after the loop
        MASK_VEC = ~MASK_TAIL;

        __m512 sum = _mm512_set1_ps(0.0f);

        for (size_t cohort = 0; cohort < (M & MASK_VEC); cohort += N_PACKED_FLOATS) {
            // profit[i] += revenue[i * M + cohort];
            // but do N_PACKED_FLOATS iterations in parallel
            __m512 rev = _mm512_loadu_ps(&revenue[i * M + cohort]);
            sum = _mm512_add_ps(sum, rev);
        }

        // tail, add remaining elements
        size_t remainder = M % N_PACKED_FLOATS;
        if (remainder) {
            uint16_t m = 0xFFFFu >> (N_PACKED_FLOATS - remainder);
            auto mask = _load_mask16(&m);
            __m512 rev = _mm512_loadu_ps(&revenue[i * M + (M & MASK_VEC)]);
            sum = _mm512_mask_add_ps(sum, mask, sum, rev);
        }

        // sum is in vector form,
        profit[i] += _mm512_reduce_add_ps(sum) - cost[i];
    }
}


template<>
__always_inline void
reduceRevenue<Policy::PARALLEL_OMP>(const Tensor<float> &revenue, const Tensor<float> &cost, Tensor<float> &profit,
                                    size_t N, size_t M, size_t geos, size_t teams) {
#pragma omp parallel for
    for (size_t i = 0; i < geos * teams * N; ++i) {
        for (size_t cohort = 0; cohort < M; ++cohort) {
            profit[i] += revenue[i * M + cohort];
        }
        profit[i] -= cost[i];
    }
}
#pragma clang diagnostic pop
