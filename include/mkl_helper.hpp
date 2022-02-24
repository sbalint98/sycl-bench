#include<type_traits>
#include<iostream>
#include<oneapi/mkl.hpp>

template <typename fp>
typename std::enable_if<!std::is_integral<fp>::value, bool>::type check_equal(fp x, fp x_ref,
                                                                              int error_mag) {
    //using fp_real = typename complex_info<fp>::real_type;
    float bound = (error_mag * 1 * std::numeric_limits<float>::epsilon());

    bool ok;

    float aerr = std::abs(x - x_ref);
    float rerr = aerr / std::abs(x_ref);
    ok = (rerr <= bound) || (aerr <= bound);
    if (!ok)
        std::cout << "relative error = " << rerr << " absolute error = " << aerr
                  << " limit = " << bound << std::endl;
    return ok;
}

template <typename fp>
typename std::enable_if<std::is_integral<fp>::value, bool>::type check_equal(fp x, fp x_ref,
                                                                             int error_mag) {
    return (x == x_ref);
}

template <typename Fp, typename AllocType>
static inline bool check_equal_vector(std::vector<Fp, AllocType>& r1,
                                      std::vector<Fp, AllocType>& r2) {
    bool good = true;
    for (int i = 0; i < r1.size(); i++) {
        if (!check_equal(r1[i], r2[i])) {
            good = false;
            break;
        }
    }
    return good;
}

template <typename fp>
bool check_equal_vector(fp *v, fp *v_ref, int n, int inc, int error_mag, std::ostream &out) {
    int abs_inc = std::abs(inc), count = 0;
    bool good = true;

    for (int i = 0; i < n; i++) {
        if (!check_equal(v[i * abs_inc], v_ref[i * abs_inc], error_mag)) {
            int i_actual = (inc > 0) ? i : n - i;
            std::cout << "Difference in entry " << i_actual << ": DPC++ " << v[i * abs_inc]
                      << " vs. Reference " << v_ref[i * abs_inc] << std::endl;
            good = false;
            count++;
            if (count > 100)
                return good;
        }
    }

    return good;
}


// Random initialization.
template <typename fp>
static fp rand_scalar() {
    return fp(std::rand()) / fp(RAND_MAX) - fp(0.5);
}
template <typename fp>
static std::complex<fp> rand_complex_scalar() {
    return std::complex<fp>(rand_scalar<fp>(), rand_scalar<fp>());
}
template <>
std::complex<float> rand_scalar() {
    return rand_complex_scalar<float>();
}
template <>
std::complex<double> rand_scalar() {
    return rand_complex_scalar<double>();
}
template <>
int8_t rand_scalar() {
    return std::rand() % 254 - 127;
}
template <>
int32_t rand_scalar() {
    return std::rand() % 256 - 128;
}
template <>
uint8_t rand_scalar() {
    return std::rand() % 128;
}

template <>
sycl::half rand_scalar() {
    return sycl::half(std::rand() % 32000) / sycl::half(32000) - sycl::half(0.5);
}

template <typename fp>
static fp rand_scalar(int mag) {
    fp tmp = fp(mag) + fp(std::rand()) / fp(RAND_MAX) - fp(0.5);
    if (std::rand() % 2)
        return tmp;
    else
        return -tmp;
}

// Matrix helpers.
template <typename T>
constexpr T inner_dimension(oneapi::mkl::transpose trans, T m, T n) {
    return (trans == oneapi::mkl::transpose::nontrans) ? m : n;
}
template <typename T>
constexpr T outer_dimension(oneapi::mkl::transpose trans, T m, T n) {
    return (trans == oneapi::mkl::transpose::nontrans) ? n : m;
}

template <typename T>
constexpr T matrix_size(oneapi::mkl::transpose trans, T m, T n, T ldm) {
    return outer_dimension(trans, m, n) * ldm;
}
template <typename T>
constexpr T matrix_size(oneapi::mkl::layout layout, oneapi::mkl::transpose trans, T m, T n, T ldm) {
    return (layout == oneapi::mkl::layout::column_major) ? outer_dimension(trans, m, n) * ldm
                                                         : inner_dimension(trans, m, n) * ldm;
}

template <typename vec>
void rand_vector(vec &v, int n, int inc) {
    using fp = typename vec::value_type;
    int abs_inc = std::abs(inc);

    v.resize(n * abs_inc);

    for (int i = 0; i < n; i++)
        v[i * abs_inc] = rand_scalar<fp>();
}

template <typename fp>
void rand_vector(fp *v, int n, int inc) {
    int abs_inc = std::abs(inc);
    for (int i = 0; i < n; i++)
        v[i * abs_inc] = rand_scalar<fp>();
}


template <typename vec>
void rand_matrix(vec &M, oneapi::mkl::transpose trans, int m, int n, int ld) {
    using fp = typename vec::value_type;

    M.resize(matrix_size(trans, m, n, ld));

    if (trans == oneapi::mkl::transpose::nontrans) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                M[i + j * ld] = rand_scalar<fp>();
    }
    else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                M[j + i * ld] = rand_scalar<fp>();
    }
}

template <typename fp>
void rand_matrix(fp *M, oneapi::mkl::layout layout, oneapi::mkl::transpose trans, int m, int n,
                 int ld) {
    if (((trans == oneapi::mkl::transpose::nontrans) &&
         (layout == oneapi::mkl::layout::column_major)) ||
        ((trans != oneapi::mkl::transpose::nontrans) &&
         (layout == oneapi::mkl::layout::row_major))) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                M[i + j * ld] = rand_scalar<fp>();
    }
    else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                M[j + i * ld] = rand_scalar<fp>();
    }
}

inline CBLAS_LAYOUT convert_to_cblas_layout(oneapi::mkl::layout is_column) {
    return is_column == oneapi::mkl::layout::column_major ? CBLAS_LAYOUT::CblasColMajor
                                                          : CBLAS_LAYOUT::CblasRowMajor;
}

inline CBLAS_TRANSPOSE convert_to_cblas_trans(oneapi::mkl::transpose trans) {
    if (trans == oneapi::mkl::transpose::trans)
        return CBLAS_TRANSPOSE::CblasTrans;
    else if (trans == oneapi::mkl::transpose::conjtrans)
        return CBLAS_TRANSPOSE::CblasConjTrans;
    else
        return CBLAS_TRANSPOSE::CblasNoTrans;
}