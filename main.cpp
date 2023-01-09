#include <iostream>
#include "Tensor.h"
#include "reduceRevenue.h"

#include <chrono>
#include <cstdint>

#ifndef POLICY
#error POLICY macro must be defined
#endif

using std::chrono::duration_cast;

int main() {
    // TODO cli params
    constexpr size_t NTrials = 10;

    size_t N = 2000, M = N, geos = 4, teams = 5;
//    size_t N = 5, M = N, geos = 2, teams = 1;

    // TODO std::random
    srand(0);
    Tensor<float> revenue({geos, teams, N, M}, false),
            cost({geos, teams, N}, false),
            profit({geos, teams, N});

    revenue.fill([](std::size_t index) { return (rand() % 10); });
    cost.fill([](std::size_t index) { return (rand() % 100) / 10.0f; });

//    std::cout << revenue << std::endl;
//    std::cout << cost << std::endl;
//    std::cout << profit << std::endl;
    std::chrono::nanoseconds duration[NTrials];

    for (auto &d: duration) {
        profit.fill([](std::size_t i){ return 0; });

        auto start = std::chrono::high_resolution_clock::now();

        reduceRevenue<Policy::POLICY>(revenue, cost, profit, N, M, geos, teams);
        std::cout << profit({0, 0, 0, 0}) << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        d = end - start;
        std::cout << duration_cast<std::chrono::duration<float, std::milli>>(d).count() << "ms" << std::endl;
    }

    Tensor<float> profit2({geos, teams, N});
    reduceRevenue<Policy::SERIAL_FAST>(revenue, cost, profit2, N, M, geos, teams);

    assert(profit.size() == profit2.size());
    for (int i = 0; i < profit2.size(); ++i) {
        assert(profit[i] == profit2[i]);
    }

    for (auto d: duration) {

    }

//    std::cout << revenue << std::endl;
//    std::cout << cost << std::endl;
//    std::cout << profit << std::endl;

    return 0;
}
