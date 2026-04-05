#include <array>
#include <vector>

using Mat4 = std::array<std::array<double, 4>, 4>;
using Mat2 = std::array<std::array<double, 2>, 2>;
using Vec4 = std::array<double, 4>;
using Vec2 = std::array<double, 2>;
using Mat4x2 = std::array<std::array<double, 2>, 4>;
using Mat2x4 = std::array<std::array<double, 4>, 2>;

struct Measurement {
    double t;        // time (seconds)
    double range;    // meters
    double bearing;  // radians
};

class Ekf {
public:
    Ekf(Measurement first);
    ~Ekf() = default;

    void predict();
    void update(const Measurement& measurement);

    Vec4 xhat;
    Mat4 I;
    Mat4 F;
    Mat4 F_T; // F transposed
    Mat2x4 H;
    Mat4x2 H_T; // H transposed
    Mat4 P;
    Mat4x2 K;
    Vec2 z;
    Vec2 zhat;
    Vec2 y;
    const double dt{0.5};
    const double sigma_a{8}; // random acceleration process noise m/s^2
    Mat2 R;
    Mat4 Q;

    void prop_state();
    void prop_covariance();
    Vec2 state_to_meas(const Vec4&);
    void compute_meas_predict();
    void compute_residual();
    void compute_meas_jacobian();
    void compute_K_gain();
    void update_state();
    void update_covariance();

    Vec4 mult_M4_Vec4(const Mat4&, const Vec4&);
    Mat4 mult_M4(const Mat4&, const Mat4&);
    Mat4 add_M4(const Mat4&, const Mat4&);
    Mat4 sub_M4(const Mat4&, const Mat4&);
    Mat2 mult_M2(const Mat2&, const Mat2&);
    Mat2 add_M2(const Mat2&, const Mat2&);
    Vec2 sub_V2(const Vec2&, const Vec2&);
    Mat2 mult_M2x4_M4x2(const Mat2x4&, const Mat4x2&);
    Mat4x2 mult_M4x4_M4x2(const Mat4&, const Mat4x2&);
    Mat4x2 mult_M4x2_M2x2(const Mat4x2&, const Mat2&);
    Mat4 mult_M4x2_M2x4(const Mat4x2&, const Mat2x4&);
    Vec4 mult_M4x2_V2(const Mat4x2&, const Vec2&);
    Vec4 add_V4(const Vec4&, const Vec4&);
    Mat4x2 transpose_M2x4(const Mat2x4&);
    Mat2 invert_M2(const Mat2&);
};