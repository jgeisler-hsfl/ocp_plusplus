// ocp/solvers/acados/qpscaling.hpp
//
// QP scaling: produces a numerically well-conditioned copy of an
// assembled staged QP for the QP solver and maps the solved
// (scaled-space) solution back to original space.
//
// Two independent scalings, both applied in-place to the copy:
//   - Objective scaling: a single scalar factor derived from the
//     Gershgorin max-abs-eigenvalue estimate of the stage Hessians and
//     the inf-norm of the stage gradients.
//   - Constraint scaling: a per-row factor
//     s_j = 1/max(1, max(bound_max, coeff_norm)) for every
//     general-constraint row (ineq/eq/lin); the (u;x) Jacobian row and
//     rhs are multiplied by s_j, and for soft rows the slack Hessian
//     diagonal is divided by s_j^2 and the slack gradient by s_j.
//     Box and pin rows are not scaled.
//
// Supplementary - correspondence with the HPIPM/acados
// implementation:
//   QpScaler                   acados/ocp_nlp/ocp_nlp_qpscaling.c
//   compute_obj_scaling_factor acados/ocp_nlp/ocp_nlp_qpscaling.c:483-540
//   scale_stage_general_       acados/ocp_nlp/ocp_nlp_qpscaling.c:545-618
//   rescale_stage_             acados/ocp_nlp/ocp_nlp_qpscaling.c:327-375
//   gershgorin_max_abs_eig     acados/utils/math.c:1118
//
// HPIPM reference:
//   acados/ocp_nlp/ocp_nlp_qpscaling.c
//   acados/utils/math.c

#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "../hpipm/qp.hpp"
#include "../../utils/inspect.hpp"

namespace ocp
{

/// QP scaling.
///
/// Scales the staged QP into the well-conditioned copy `scaled_in_` for
/// the QP solver and maps the solved (scaled-space) QpSol back to
/// original space via `rescale_solution()`.
///
/// Objective scaling: a single scalar factor determined by the
/// Gershgorin max-abs-eigenvalue estimate of every stage Hessian and the
/// inf-norm of the gradients.
/// Constraint scaling: each general-constraint row (ineq/eq/lin) gets a
/// per-row factor s_j = 1/max(1, max(bound_max, coeff_norm)); the
/// (u;x) Jacobian row and rhs are multiplied by s_j, and for soft rows
/// the slack Hessian diagonal is divided by s_j^2 and the slack gradient
/// by s_j.
///
/// Box and pin rows are NOT scaled.
template <class P, int NH = Eigen::Dynamic>
class QpScaler
{
public:
    using D = QpDim<P>;
    using S = typename P::scalar_t;

    /// Scaling options (both scalings can be disabled independently).
    struct Opts
    {
        bool scale_objective = false;        // enable objective scaling
        bool scale_constraints = false;      // enable constraint scaling
        double ub_max_abs_eig = 1e5;         // cap for max-abs-eig estimate
        double lb_norm_inf_grad_obj = 1e-4;  // floor for scaled grad inf-norm

        /// By-name access (docs/plans/OPTION_INSPECT_PLAN.md): linked under
        /// "options" by QpScaler::inspect (paths scaling.options.*).
        template <class Visitor>
        static constexpr void inspect(Visitor&& vis)
        {
            vis("scale_objective", &Opts::scale_objective);
            vis("scale_constraints", &Opts::scale_constraints);
            // ub_max_abs_eig must stay > 0: it feeds the objective scale
            // factor, which the rescale divides by (1 / obj_factor).
            vis("ub_max_abs_eig", &Opts::ub_max_abs_eig,
                inspect::kMinPositive, std::nullopt);
            vis("lb_norm_inf_grad_obj", &Opts::lb_norm_inf_grad_obj,
                0.0, std::nullopt);
        }
    };

    explicit QpScaler(Opts opts = {}) : opts_(opts)
    {
    }

    /// Replace the scaling options between uses (e.g. from an interactive
    /// environment). The factors are re-derived on the next scale_qp().
    void set_opts(const Opts& o)
    {
        opts_ = o;
    }

    /// inspect children (docs/plans/OPTION_INSPECT_PLAN.md sec. 3): the live
    /// scaling options (by-name paths: scaling.options.*).
    template <class Visitor>
    static constexpr void inspect(Visitor&& vis)
    {
        vis("options", &QpScaler::opts_);
    }

    /// @param n_stages number of stages N
    void resize(int n_stages)
    {
        scaled_in_.resize(n_stages);
        scaled_out_.resize(n_stages);
    }

    /// Scale the assembled (original-space) QP into scaled_in_.
    ///
    /// With neither scaling active, scaled_in_ is a plain copy and
    /// obj_factor() / min_constr_scaling() stay 1.0.
    /// @param in assembled staged QP in original space
    void scale_qp(const Qp<P, NH>& in)
    {
        scaled_in_ = in;
        if (opts_.scale_objective)
        {
            obj_factor_ = compute_obj_scaling_factor(
                scaled_in_, opts_.ub_max_abs_eig,
                opts_.lb_norm_inf_grad_obj);
            scale_objective_(scaled_in_, obj_factor_);
        }
        else
        {
            obj_factor_ = 1.0;
        }
        min_constr_scaling_ = 1.0;
        if (opts_.scale_constraints)
        {
            scale_constraints_(scaled_in_);
        }
    }

    /// Map the solved (scaled-space) solution back to original space.
    ///
    /// Multiplies dual multipliers by 1/obj_factor (objective scaling)
    /// and by the per-row constraint factors, and divides the solved
    /// slack variables by the per-row factors (constraint scaling).
    /// @param[out] out solution in original space
    void rescale_solution(QpSol<P, NH>& out) const
    {
        out = scaled_out_;
        if (opts_.scale_objective)
        {
            const S inv = static_cast<S>(1.0 / obj_factor_);
            for (int k = 0; k < out.N; ++k)
            {
                out.pi[k] *= inv;
            }
            out.lam_first *= inv;
            for (int k = 1; k < out.N; ++k)
            {
                out.lam_path[k - 1] *= inv;
            }
            out.lam_term *= inv;
        }
        if (opts_.scale_constraints)
        {
            rescale_stage_(out.lam_first, out.ux_first, D::lay_first,
                           D::nu + D::nx, D::idxs_lo_first,
                           D::idxs_hi_first, cs_first_);
            for (int k = 1; k < out.N; ++k)
            {
                rescale_stage_(out.lam_path[k - 1], out.ux_path[k - 1],
                               D::lay_path, D::nu + D::nx,
                               D::idxs_lo_path, D::idxs_hi_path,
                               cs_path_);
            }
            rescale_stage_(out.lam_term, out.ux_term, D::lay_term, D::nx,
                           D::idxs_lo_term, D::idxs_hi_term, cs_term_);
        }
    }

    Qp<P, NH>& scaled_in() { return scaled_in_; }
    const Qp<P, NH>& scaled_in() const { return scaled_in_; }
    QpSol<P, NH>& scaled_out() { return scaled_out_; }
    const QpSol<P, NH>& scaled_out() const { return scaled_out_; }
    double obj_factor() const { return obj_factor_; }
    double min_constr_scaling() const { return min_constr_scaling_; }

    /// Objective scaling factor.
    ///
    /// 1 while the Gershgorin max-abs-eigenvalue estimate `e` of the
    /// stage Hessians is below ub_max_abs_eig, and
    /// ub_max_abs_eig / e otherwise; raised (capped by the upscale
    /// bound) if the scaled gradient inf-norm would fall below
    /// lb_norm_inf_grad_obj. Static so it can be unit-tested without a
    /// QpScaler instance.
    /// @param  qp staged QP whose Hessians and gradients are inspected
    /// @param  ub_max_abs_eig upper bound for the max-abs-eig estimate
    /// @param  lb_norm_inf_grad_obj lower bound for the scaled
    ///         gradient inf-norm
    /// @return objective scaling factor
    static double compute_obj_scaling_factor(const Qp<P, NH>& qp,
                                             double ub_max_abs_eig,
                                             double lb_norm_inf_grad_obj)
    {
        double max_abs_eig = 0.0;
        double grad_norm = 0.0;

        // Stage order: first (k = 0), path (k = 0..N-2), terminal
        // (k = N-1). The max-abs-eig estimate covers the (u;x) Hessian
        // block plus the diagonal slack weights Z: since the slack block
        // is diagonal and decoupled, the Gershgorin bound over the full
        // (u;x;s) block equals max(gershgorin(H_ux), max(Z)).
        max_abs_eig = std::max(max_abs_eig,
                               gershgorin_max_abs_eig(qp.first.hess));
        grad_norm = std::max(grad_norm, inf_norm(qp.first.grad));
        for (int k = 0; k < qp.N; ++k)
        {
            if (k < qp.N - 1)
            {
                max_abs_eig = std::max(
                    max_abs_eig,
                    gershgorin_max_abs_eig(qp.path[k].hess));
                grad_norm = std::max(grad_norm,
                                     inf_norm(qp.path[k].grad));
            }
            else
            {
                max_abs_eig = std::max(
                    max_abs_eig,
                    gershgorin_max_abs_eig(qp.term.hess));
                grad_norm = std::max(grad_norm,
                                     inf_norm(qp.term.grad));
            }
        }

        const double e = std::max(max_abs_eig, 1e-300);
        double obj_factor;
        double max_upscale;
        if (max_abs_eig < ub_max_abs_eig)
        {
            obj_factor = 1.0;
            max_upscale = ub_max_abs_eig / e;
        }
        else
        {
            obj_factor = ub_max_abs_eig / e;
            max_upscale = obj_factor;
        }
        if (grad_norm > 1e-300 &&
            obj_factor * grad_norm <= lb_norm_inf_grad_obj)
        {
            const double lb_factor =
                lb_norm_inf_grad_obj / grad_norm;
            obj_factor =
                std::max(obj_factor,
                         std::min(max_upscale, lb_factor));
        }
        return obj_factor;
    }

private:
    static constexpr int ngen_first = D::ng + D::ne + D::nl;
    static constexpr int ngen_path = D::ng + D::ne + D::nl;
    static constexpr int ngen_term = D::ng_t + D::ne_t + D::nl_t;

    /// Gershgorin max-abs-eigenvalue estimate of a symmetric matrix:
    /// max_i max(|a_ii - r_i|, |a_ii + r_i|) with
    /// r_i = sum_{j != i} |a_ij|.
    /// @param  A symmetric matrix (or block expression of one)
    /// @return upper bound on the magnitude of the largest eigenvalue
    template <class H>
    static double gershgorin_max_abs_eig(const H& A)
    {
        const int n = static_cast<int>(A.rows());
        double m = 0.0;
        for (int i = 0; i < n; ++i)
        {
            double r = 0.0;
            for (int j = 0; j < n; ++j)
            {
                if (j != i)
                {
                    r += std::fabs(static_cast<double>(A(i, j)));
                }
            }
            const double a = static_cast<double>(A(i, i));
            m = std::max(m, std::max(std::fabs(a - r),
                                     std::fabs(a + r)));
        }
        return m;
    }

    /// Infinity norm (max absolute entry) of a vector.
    /// @param  v vector (or vector expression)
    /// @return max_i |v_i|
    template <class V>
    static double inf_norm(const V& v)
    {
        double m = 0.0;
        for (int i = 0; i < v.size(); ++i)
        {
            m = std::max(m,
                         std::fabs(static_cast<double>(v(i))));
        }
        return m;
    }

    /// Multiply every stage Hessian and gradient by `f` (first stage,
    /// path stages, terminal stage).
    /// @param[out] qp staged QP, modified in place
    /// @param f objective scaling factor
    void scale_objective_(Qp<P, NH>& qp, double f) const
    {
        const S sf = static_cast<S>(f);
        qp.first.hess *= sf;
        qp.first.grad *= sf;
        for (int k = 0; k < qp.N - 1; ++k)
        {
            qp.path[k].hess *= sf;
            qp.path[k].grad *= sf;
        }
        qp.term.hess *= sf;
        qp.term.grad *= sf;
    }

    /// Per-row scaling of the general-constraint rows of every stage.
    /// @param[out] qp staged QP, modified in place
    void scale_constraints_(Qp<P, NH>& qp)
    {
        scale_stage_general_(qp.first, D::lay_first, D::nu + D::nx,
                             D::idxs_lo_first, D::idxs_hi_first,
                             cs_first_);
        for (int k = 0; k < qp.N - 1; ++k)
        {
            scale_stage_general_(qp.path[k], D::lay_path,
                                 D::nu + D::nx,
                                 D::idxs_lo_path, D::idxs_hi_path,
                                 cs_path_);
        }
        scale_stage_general_(qp.term, D::lay_term, D::nx,
                             D::idxs_lo_term, D::idxs_hi_term,
                             cs_term_);
    }

    /// Per-row constraint scaling for one stage.
    ///
    /// Only general rows (ineq/eq/lin) are scaled; box/pin are untouched.
    /// Row r gets s_r = 1/max(1, max(|bound|, |coeff|)); the (u;x)
    /// Jacobian row and rhs are multiplied by s_r, and for soft rows the
    /// slack Hessian diagonal is divided by s_r^2 and the slack gradient
    /// by s_r. The smallest factor is tracked in min_constr_scaling_.
    /// @param[out] st stage data, modified in place
    /// @param lay row/side layout of the stage
    /// @param nux size of the combined (u;x) vector
    /// @param idxs_lo per-row index of the lower slack (-1: none)
    /// @param idxs_hi per-row index of the upper slack (-1: none)
    /// @param[out] cs per-row scaling factors, stored
    template <class St, class Idx, class Cs>
    void scale_stage_general_(St& st, const detail::QpLayout& lay,
                              int nux, const Idx& idxs_lo,
                              const Idx& idxs_hi, Cs& cs)
    {
        const int gbase = lay.row_off(detail::g_ineq);
        const int groups[3] = {
            detail::g_ineq, detail::g_eq, detail::g_lin};
        for (int gi = 0; gi < 3; ++gi)
        {
            const int g = groups[gi];
            const int nrm = lay.rows[g];
            const int roff = lay.row_off(g);
            for (int j = 0; j < nrm; ++j)
            {
                const int r = roff + j;
                double coeff = 0.0;
                for (int c = 0; c < nux; ++c)
                {
                    coeff = std::max(
                        coeff,
                        std::fabs(static_cast<double>(
                            st.DC(r, c))));
                }
                const int slo = lay.side_lo(r);
                const int shi = lay.side_hi(r);
                const double mlo =
                    (slo >= 0) ? static_cast<double>(
                                     st.d_mask(slo)) : 0.0;
                const double dlo =
                    (slo >= 0) ? static_cast<double>(
                                     st.d(slo)) : 0.0;
                const double mhi =
                    static_cast<double>(st.d_mask(shi));
                const double dhi =
                    static_cast<double>(st.d(shi));
                const double bound = std::max(
                    std::fabs(mlo * dlo),
                    std::fabs(mhi * dhi));
                const double s =
                    1.0 / std::max(1.0, std::max(bound, coeff));
                cs[r - gbase] = s;
                min_constr_scaling_ =
                    std::min(min_constr_scaling_, s);

                for (int c = 0; c < nux; ++c)
                {
                    st.DC(r, c) *= static_cast<S>(s);
                }
                if (slo >= 0)
                {
                    st.d(slo) *= static_cast<S>(s);
                }
                st.d(shi) *= static_cast<S>(s);

                // Soft-row slack cost rescale: keep the slack penalty
                // consistent with the scaled constraint row,
                // Z_slack /= s^2, rqz_slack /= s.
                const int clo = idxs_lo[r];
                const int chi = idxs_hi[r];
                if (clo >= 0)
                {
                    st.hess(clo, clo) *=
                        static_cast<S>(1.0 / (s * s));
                    st.grad(clo) *= static_cast<S>(1.0 / s);
                }
                if (chi >= 0)
                {
                    st.hess(chi, chi) *=
                        static_cast<S>(1.0 / (s * s));
                    st.grad(chi) *= static_cast<S>(1.0 / s);
                }
            }
        }
    }

    /// Rescale the solved multipliers and slacks of one stage by the
    /// per-row constraint factors (inverse of scale_stage_general_).
    /// @param[out] lam solved multipliers, modified in place
    /// @param[out] ux solved stage vector (u;x) followed by slacks,
    ///           modified in place
    /// @param lay row/side layout of the stage
    /// @param nv0 size of the leading (u;x) part of the stage vector
    /// @param idxs_lo per-row index of the lower slack (-1: none)
    /// @param idxs_hi per-row index of the upper slack (-1: none)
    /// @param cs per-row scaling factors from scale_stage_general_
    template <class Lam, class Ux, class Idx, class Cs>
    void rescale_stage_(Lam& lam, Ux& ux, const detail::QpLayout& lay,
                        int nv0, const Idx& idxs_lo,
                        const Idx& idxs_hi,
                        const Cs& cs) const
    {
        const int gbase = lay.row_off(detail::g_ineq);
        const int groups[3] = {
            detail::g_ineq, detail::g_eq, detail::g_lin};
        const int ssbase = lay.lo_size() + lay.nrow();
        for (int gi = 0; gi < 3; ++gi)
        {
            const int g = groups[gi];
            const int nrm = lay.rows[g];
            const int roff = lay.row_off(g);
            for (int j = 0; j < nrm; ++j)
            {
                const int r = roff + j;
                const double s = cs[r - gbase];
                const S sf = static_cast<S>(s);
                const int shi = lay.side_hi(r);
                lam(shi) *= sf;
                const int slo = lay.side_lo(r);
                if (slo >= 0)
                {
                    lam(slo) *= sf;
                }
                const int clo = idxs_lo[r];
                const int chi = idxs_hi[r];
                if (clo >= 0)
                {
                    ux(clo) /= sf;
                    lam(ssbase + (clo - nv0)) *= sf;
                }
                if (chi >= 0)
                {
                    ux(chi) /= sf;
                    lam(ssbase + (chi - nv0)) *= sf;
                }
            }
        }
    }

    Qp<P, NH> scaled_in_;                 // scaled copy of the QP
    QpSol<P, NH> scaled_out_;             // solution buffer (scaled space)
    Opts opts_;
    double obj_factor_ = 1.0;             // applied objective scaling factor
    double min_constr_scaling_ = 1.0;     // smallest per-row constraint factor
    std::array<double, ngen_first> cs_first_{};  // row factors, first stage
    std::array<double, ngen_path> cs_path_{};    // row factors, path stages
    std::array<double, ngen_term> cs_term_{};    // row factors, terminal
};

}  // namespace ocp
