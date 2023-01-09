#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
#pragma once

#include <cilk/cilk.h>

enum class Policy {
    SERIAL,
    SERIAL_VEC,
    SERIAL_FAST,
    PARALLEL,
    PARALLEL_OMP,
    LENGTH
};

///
/// \param N number of time steps
/// \param M number of cohorts
/// \param geos
/// \param teams
/// \return revenue, cost, profit
__always_inline std::array<Tensor<float>, 3> initialize(size_t N, size_t M, size_t geos, size_t teams) {
    Tensor<float> revenue({geos, teams, N, M}, false), cost({geos, teams, N}, false), profit({geos, teams, N});

    for (size_t i = 0; i < revenue.size(); ++i) {
        revenue[i] = (rand() % 10); // 10.0f;
    }

    for (size_t i = 0; i < cost.size(); ++i) {
        cost[i] = (rand() % 100) / 10.0f;
    }

    return {std::move(revenue), std::move(cost), std::move(profit)};
}

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
reduceRevenue<Policy::SERIAL_VEC>(const Tensor<float> &revenue, const Tensor<float> &cost, Tensor<float> &profit,
                              size_t N, size_t M, size_t geos, size_t teams) {
    for (size_t geo = 0; geo < geos; ++geo) {
        for (size_t team = 0; team < teams; ++team) {
            for (size_t time = 0; time < N; ++time) {
                for (size_t cohort = 0; cohort < M; ++cohort) {
                    profit({geo, team, time}) += revenue({geo, team, time, cohort});
                }
                profit({geo, team, time}) -= cost({geo, team, time});
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
reduceRevenue<Policy::PARALLEL>(const Tensor<float> &revenue, const Tensor<float> &cost, Tensor<float> &profit,
                                size_t N, size_t M, size_t geos, size_t teams) {
    cilk_for (size_t i = 0; i < geos * teams * N; ++i) {
        for (size_t cohort = 0; cohort < M; ++cohort) {
            profit[i] += revenue[i * M + cohort];
        }
        profit[i] -= cost[i];
    }
}


template<>
__always_inline void
reduceRevenue<Policy::PARALLEL_OMP>(const Tensor<float> &revenue, const Tensor<float> &cost, Tensor<float> &profit,
                                    size_t N, size_t M, size_t geos, size_t teams) {
#pragma omp parallel for
    for (size_t geo = 0; geo < geos; ++geo) {
        for (size_t team = 0; team < teams; ++team) {
            for (size_t time = 0; time < N; ++time) {
                for (size_t cohort = 0; cohort < M; ++cohort) {
                    profit({geo, team, time}) += revenue({geo, team, time, cohort});
                }
                profit({geo, team, time}) -= cost({geo, team, time});
            }
        }
    }
}
#pragma clang diagnostic pop
