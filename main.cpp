#include <iostream>
#include "Tensor.h"
#include "reduceRevenue.h"

#include <chrono>
#include <cstdint>
#include <random>
#include <argparse/argparse.hpp>

#ifndef POLICY
#error POLICY macro must be defined
#endif

using std::chrono::duration_cast;

int main(int argc, const char *argv[]) {
    // TODO cli params
    constexpr size_t NTrials = 10;

    argparse::ArgumentParser args{"causal"};
    args.add_argument("-N").default_value(2000ul).scan<'u', size_t>();
    args.add_argument("-M").help("If set to 0, it'll be equal to N").default_value(0ul).scan<'u', size_t>();
    args.add_argument("-g").default_value(4ul).scan<'u', size_t>();
    args.add_argument("-t").default_value(5ul).scan<'u', size_t>();

    args.parse_args(argc, argv);

    auto N = args.get<size_t>("-N"),
            M = args.get<size_t>("-M"),
            geos = args.get<size_t>("-g"),
            teams = args.get<size_t>("-t");

    if (M == 0) M = N;

    std::mt19937 random{0};
    Tensor<float> revenue({geos, teams, N, M}, false),
            cost({geos, teams, N}, false),
            profit({geos, teams, N});

    revenue.fill([&random](std::size_t index) { return (random() % 10); });
    cost.fill([&random](std::size_t index) { return static_cast<float>(random() % 100) / 10.0f; });

//    std::cout << revenue << std::endl;
//    std::cout << cost << std::endl;
//    std::cout << profit << std::endl;
    std::array<std::chrono::duration<float, std::milli>, NTrials> duration{};

    for (auto &d: duration) {
        profit.fill([](std::size_t i) { return 0; });

        auto start = std::chrono::high_resolution_clock::now();

        reduceRevenue<Policy::POLICY>(revenue, cost, profit, N, M, geos, teams);
//        std::cout << profit({0, 0, 0}) << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        d = end - start;
        std::cout << d.count() << "ms" << std::endl;
    }

    Tensor<float> profit2({geos, teams, N});
    reduceRevenue<Policy::SERIAL_FAST>(revenue, cost, profit2, N, M, geos, teams);

    assert(profit.size() == profit2.size());
    for (int i = 0; i < profit2.size(); ++i) {
        assert(profit[i] == profit2[i]);
    }

    // collect execution stats
    std::sort(duration.begin(), duration.end());
    auto avg = std::reduce(duration.begin(), duration.end()) / NTrials;
    decltype(avg) median;
    if (NTrials % 2 == 0) {
        median = (duration[NTrials / 2] + duration[NTrials / 2 - 1]) / 2;
    } else {
        median = duration[NTrials / 2];
    }

    std::cout << "Minimum: " << duration[0].count() << "ms" << std::endl
              << "Average: " << avg.count() << "ms" << std::endl
              << "Median: " << median.count() << "ms" << std::endl;

//    std::cout << revenue << std::endl;
//    std::cout << cost << std::endl;
//    std::cout << profit << std::endl;
//    std::cout << profit2 << std::endl;

    return 0;
}
