#pragma once

#include <cstdint>
#include <cstring>
#include <span>
#include <algorithm>
#include <cassert>
#include <ostream>
#include <concepts>

using std::size_t;

template<class T>
concept is_size_t = std::same_as<T, size_t>;

template<class T>
struct Tensor;

template<class T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor);

template<class T>
struct Tensor {

    explicit Tensor(std::vector<size_t> ns, bool zero = true) : ns(std::move(ns)) {
        total = 1;
        for (auto n: this->ns) {
            total *= n;
        }

        data = new T[total];
        if (zero) {
            memset(data, 0, total * sizeof(T));
        }
    }

    /// first coordinate is the highest-order (last coordinates are in a row)
    T &operator()(const std::vector<size_t> &coordinates) {
        return data[rawIndex(coordinates)];
    }

    const T &operator()(const std::vector<size_t> &coordinates) const {
        return data[rawIndex(coordinates)];
    }

    template<std::same_as<size_t> ...Args>
    T &operator()(Args ...args) {
        return (*this)({args...});
    }

    template<std::same_as<size_t> ...Args>
    const T &operator()(Args ...args) const {
        return (*this)({args...});
    }

    T &operator[](size_t index) {
        return data[index];
    }

    const T &operator[](size_t index) const {
        return data[index];
    }

    [[nodiscard]] size_t rawIndex(const std::vector<size_t> &coordinates) const {
        assert(coordinates.size() == ns.size());
        size_t index = 0;

        for (size_t i = 0; i < ns.size(); i++) {
            index = index * ns[i] + coordinates[i];
        }
        assert(index < total);
        return index;
    }

    /// for computation, just so we don't make *everything* a member function
    [[nodiscard]] T *raw() {
        return data;
    }

    [[nodiscard]] const T *raw() const {
        return data;
    }

    [[nodiscard]] size_t size() const {
        return total;
    }

    // ====================
    // Constructors
    // ====================

    Tensor(const Tensor &other) : ns(other.ns), total(other.total) {
        data = new T[total];
        memcpy(data, other.data, total * sizeof(T));
    }

    // TODO warning
    Tensor &operator=(const Tensor &other) {
        delete[] data;

        ns = other.ns;
        total = other.total;
        data = new T[total];
        memcpy(data, other.data, total * sizeof(T));
        return *this;
    };

    Tensor(Tensor &&other) noexcept: ns(std::move(other.ns)), total(other.total), data(other.data) {
        other.data = nullptr;
    }

    Tensor &operator=(Tensor &&other) {
        delete[] data;

        ns = std::move(other.ns);
        total = other.total;
        data = other.data;
        other.data = nullptr;
        return *this;
    };

    ~Tensor() {
        delete[] data;
    }

    friend std::ostream &operator<< <T>(std::ostream &os, const Tensor<T> &tensor);

private:
    std::vector<size_t> ns;
    size_t total;
    T *data;
};

template<class T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
    os << "Tensor{ns: [";
    for (auto n: tensor.ns) {
        os << n << " ";
    }

    // TODO remove last ' '
    //  os.seekp(long(os.tellp()) - 1);

    os << "] total: " << tensor.total << " data: ";
    for (int i = 0; i < tensor.total; ++i) {
        os << tensor.data[i] << " ";
    }
    os << "}";
    return os;
}
