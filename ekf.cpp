#include "ekf.h"

#include <cmath>
#include <stdexcept>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

Mat4 identity_M4() {
    Mat4 I{};
    for (int i = 0; i < 4; ++i) {
        I[i][i] = 1.0;
    }
    return I;
}

Mat2 identity_M2() {
    Mat2 I{};
    for (int i = 0; i < 2; ++i) {
        I[i][i] = 1.0;
    }
    return I;
}

Mat4 transpose_M4x4(const Mat4& A) {
    Mat4 T{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

}  // namespace

Ekf::Ekf(Measurement first)
    : xhat{},
      I(identity_M4()),
      F(identity_M4()),
      F_T(identity_M4()),
      H{},
      H_T{},
      P{},
      K{},
      z{},
      zhat{},
      y{},
      R{},
      Q{} {
    const double c = std::cos(first.bearing);
    const double s = std::sin(first.bearing);

    xhat[0] = first.range * c;
    xhat[1] = first.range * s;
    xhat[2] = 0.0;
    xhat[3] = 0.0;

    for (int i = 0; i < 4; ++i) {
        P[i][i] = (i < 2) ? 10.0 : 40.0;
    }

    R[0][0] = 5.0;
    R[0][1] = 0.0;
    R[1][0] = 0.0;
    R[1][1] = 0.08;

    const double q = sigma_a * sigma_a;
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt3 * dt;

    Q[0][0] = q * dt4 / 4.0;
    Q[0][1] = 0.0;
    Q[0][2] = q * dt3 / 2.0;
    Q[0][3] = 0.0;

    Q[1][0] = 0.0;
    Q[1][1] = q * dt4 / 4.0;
    Q[1][2] = 0.0;
    Q[1][3] = q * dt3 / 2.0;

    Q[2][0] = q * dt3 / 2.0;
    Q[2][1] = 0.0;
    Q[2][2] = q * dt2;
    Q[2][3] = 0.0;

    Q[3][0] = 0.0;
    Q[3][1] = q * dt3 / 2.0;
    Q[3][2] = 0.0;
    Q[3][3] = q * dt2;

    F[0][2] = dt;
    F[1][3] = dt;
    F_T = transpose_M4x4(F);
}

void Ekf::predict() {
    prop_state();
    prop_covariance();
}

void Ekf::update(const Measurement& measurement) {
    z[0] = measurement.range;
    z[1] = measurement.bearing;

    compute_meas_predict();
    compute_residual();
    compute_meas_jacobian();
    compute_K_gain();
    update_state();
    update_covariance();
}

void Ekf::prop_state() {
    xhat = mult_M4_Vec4(F, xhat);
}

void Ekf::prop_covariance() {
    const Mat4 PF_T = mult_M4(P, F_T);
    const Mat4 FPF_T = mult_M4(F, PF_T);
    P = add_M4(FPF_T, Q);
}

Vec2 Ekf::state_to_meas(const Vec4& x) {
    return {std::sqrt(x[0] * x[0] + x[1] * x[1]), std::atan2(x[1], x[0])};
}

void Ekf::compute_meas_predict() {
    zhat = state_to_meas(xhat);
}

void Ekf::compute_residual() {
    y = sub_V2(z, zhat);
    while (y[1] > kPi) {
        y[1] -= 2.0 * kPi;
    }
    while (y[1] < -kPi) {
        y[1] += 2.0 * kPi;
    }
}

void Ekf::compute_meas_jacobian() {
    const double px = xhat[0];
    const double py = xhat[1];
    const double r2 = px * px + py * py;
    const double r = std::sqrt(r2);

    H[0] = {px / r, py / r, 0.0, 0.0};
    H[1] = {-py / r2, px / r2, 0.0, 0.0};
    H_T = transpose_M2x4(H);
}

void Ekf::compute_K_gain() {
    const Mat4x2 PH_T = mult_M4x4_M4x2(P, H_T);
    const Mat2 HPH_T = mult_M2x4_M4x2(H, PH_T);
    const Mat2 S = add_M2(HPH_T, R);
    const Mat2 S_inv = invert_M2(S);
    K = mult_M4x2_M2x2(PH_T, S_inv);
}

void Ekf::update_state() {
    const Vec4 correction = mult_M4x2_V2(K, y);
    xhat = add_V4(xhat, correction);
}

void Ekf::update_covariance() {
    const Mat4 KH = mult_M4x2_M2x4(K, H);
    const Mat4 I_KH = sub_M4(I, KH);
    P = mult_M4(I_KH, P);
}

Vec4 Ekf::mult_M4_Vec4(const Mat4& A, const Vec4& x) {
    Vec4 y{};
    for (int i = 0; i < 4; ++i) {
        for (int k = 0; k < 4; ++k) {
            y[i] += A[i][k] * x[k];
        }
    }
    return y;
}

Mat4 Ekf::mult_M4(const Mat4& A, const Mat4& B) {
    Mat4 C{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

Mat4 Ekf::add_M4(const Mat4& A, const Mat4& B) {
    Mat4 C{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

Mat4 Ekf::sub_M4(const Mat4& A, const Mat4& B) {
    Mat4 C{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

Mat2 Ekf::mult_M2(const Mat2& A, const Mat2& B) {
    Mat2 C{};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

Mat2 Ekf::add_M2(const Mat2& A, const Mat2& B) {
    Mat2 C{};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

Vec2 Ekf::sub_V2(const Vec2& A, const Vec2& B) {
    Vec2 C{};
    for (int i = 0; i < 2; ++i) {
        C[i] = A[i] - B[i];
    }
    return C;
}

Mat2 Ekf::mult_M2x4_M4x2(const Mat2x4& A, const Mat4x2& B) {
    Mat2 C{};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 4; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

Mat4x2 Ekf::mult_M4x4_M4x2(const Mat4& A, const Mat4x2& B) {
    Mat4x2 C{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 4; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

Mat4x2 Ekf::mult_M4x2_M2x2(const Mat4x2& A, const Mat2& B) {
    Mat4x2 C{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

Mat4 Ekf::mult_M4x2_M2x4(const Mat4x2& A, const Mat2x4& B) {
    Mat4 C{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 2; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

Vec4 Ekf::mult_M4x2_V2(const Mat4x2& A, const Vec2& x) {
    Vec4 y{};
    for (int i = 0; i < 4; ++i) {
        for (int k = 0; k < 2; ++k) {
            y[i] += A[i][k] * x[k];
        }
    }
    return y;
}

Vec4 Ekf::add_V4(const Vec4& A, const Vec4& B) {
    Vec4 C{};
    for (int i = 0; i < 4; ++i) {
        C[i] = A[i] + B[i];
    }
    return C;
}

Mat4x2 Ekf::transpose_M2x4(const Mat2x4& A) {
    Mat4x2 T{};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

Mat2 Ekf::invert_M2(const Mat2& S) {
    const double det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
    if (std::abs(det) < 1e-12) {
        throw std::runtime_error("singular 2x2 matrix");
    }
    return Mat2{{
        {S[1][1] / det, -S[0][1] / det},
        {-S[1][0] / det, S[0][0] / det},
    }};
}
