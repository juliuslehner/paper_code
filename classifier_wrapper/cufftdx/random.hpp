
#ifndef CUFFTDX_EXAMPLE_RANDOM_HPP
#define CUFFTDX_EXAMPLE_RANDOM_HPP

#include <algorithm>
#include <random>
#include <vector>
#include <type_traits>

#include <cuda.h>
#include <cufftdx.hpp>

namespace example {
    template<class T>
    inline auto get_random_complex_data(size_t size, T real_min, T real_max, T imag_min, T imag_max) ->
        typename std::enable_if<std::is_floating_point<T>::value,
                                std::vector<cufftdx::make_complex_type_t<T>>>::type {
        using complex_type = cufftdx::make_complex_type_t<T>;
        std::random_device                rd;
        std::default_random_engine        gen(rd());
        std::uniform_real_distribution<T> real_distribution(real_min, real_max);
        std::uniform_real_distribution<T> imag_distribution(imag_min, imag_max);
        std::vector<complex_type>         output(size);
        std::generate(output.begin(), output.end(), [&]() {
            return complex_type {real_distribution(gen), imag_distribution(gen)};
        });
        return output;
    }

    template<class T>
    inline auto get_random_complex_data(size_t size, T real_min, T real_max, T imag_min, T imag_max) ->
        typename std::enable_if<std::is_same<T, __half>::value,
                                std::vector<cufftdx::make_complex_type_t<__half2>>>::type {
        using complex_type = cufftdx::make_complex_type_t<__half2>;
        std::random_device                    rd;
        std::default_random_engine            gen(rd());
        std::uniform_real_distribution<float> real_distribution(real_min, real_max);
        std::uniform_real_distribution<float> imag_distribution(imag_min, imag_max);
        std::vector<complex_type>             output(size);
        std::generate(output.begin(), output.end(), [&]() {
            auto xx = __float2half(real_distribution(gen));
            auto xy = __float2half(imag_distribution(gen));
            auto yx = __float2half(real_distribution(gen));
            auto yy = __float2half(imag_distribution(gen));
            auto x  = __half2 {xx, xy};
            auto y  = __half2 {yx, yy};
            return complex_type {x, y};
        });
        return output;
    }

    template<class T>
    inline auto get_random_real_data(size_t size, T min, T max) ->
        typename std::enable_if<std::is_floating_point<T>::value, std::vector<T>>::type {
        std::random_device                rd;
        std::default_random_engine        gen(rd());
        std::uniform_real_distribution<T> distribution(min, max);
        std::vector<T>                    output(size);
        std::generate(output.begin(), output.end(), [&]() {
            return distribution(gen);
        });
        return output;
    }

    template<class T>
    inline auto get_random_real_data(size_t size, float min, float max) ->
        typename std::enable_if<std::is_same<T, __half>::value, std::vector<__half2>>::type {
        std::random_device                    rd;
        std::default_random_engine            gen(rd());
        std::uniform_real_distribution<float> distribution(min, max);
        std::vector<__half2>                  output(size);
        std::generate(output.begin(), output.end(), [&]() {
            return __half2 {__half(distribution(gen)), __half(distribution(gen))};
        });
        return output;
    }
} // namespace example

#endif // CUFFTDX_EXAMPLE_RANDOM_HPP
