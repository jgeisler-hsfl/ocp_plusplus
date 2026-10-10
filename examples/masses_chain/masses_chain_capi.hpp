// masses_chain_capi.hpp
//
// C ABI wrapper around the CasADi-generated masses_chain implicit-DAE
// residual.
//
// The generated residual is
//     F(x, xdot, u) = f_expl(x, u) - xdot
// so the explicit ODE right-hand side the integrator needs is
//     f(x, u) = F(x, 0, u).
//
// Generated signature (CasADi runtime, n_in = 5, n_out = 1):
//     int masses_chain_impl_dae_fun(const real_t** arg, real_t** res,
//                                   int* iw, real_t* w, int mem);
//   arg = [ x(24); xdot(24); u(3); z(0); p(1) ]
//   res = [ F(24) ]
//   iw / w are zero-length for this model (pass null).
//
// Only the residual (masses_chain_impl_dae_fun.c) is compiled into the
// ocp++ target. The generated derivative routines
// (masses_chain_impl_dae_fun_jac_x_xdot_u.c, ...) are kept in the tree but
// not used: this wrapper's consumer supplies central finite-difference
// Jacobians instead.

#pragma once

namespace ocp
{

namespace detail
{

extern "C"
{
    int masses_chain_impl_dae_fun(const double** arg, double** res,
                                  int* iw, double* w, int mem);
}

/// Evaluate the implicit-DAE residual F(x, xdot, u); with xdot = 0 this is
/// the explicit ODE right-hand side f(x, u).
/// @param  x     state (24)
/// @param  xdot  state derivative (24)
/// @param  u     control (3)
/// @param  p     OCP step (unused by the generated residual)
/// @param[out] F  residual (24)
inline void masses_chain_residual(const double* x, const double* xdot,
                                  const double* u, double p,
                                  double* F) noexcept
{
    const double* arg[5] = { x, xdot, u, /* z (0) */ nullptr, &p };
    double* res[1] = { F };
    (void)masses_chain_impl_dae_fun(arg, res, nullptr, nullptr, 0);
}

}  // namespace detail

}  // namespace ocp
