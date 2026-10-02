// ocp/solvers/acados/qpscaling.hpp
//
// QP scaling (phase 3h).
//
// Port of acados ocp_nlp_qpscaling.c: objective scaling via Gershgorin
// max-abs-eigenvalue estimate and constraint scaling via inf-norm row
// normalization, plus the rescale of the solved duals/slacks back to
// original space.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "../hpipm/qp.hpp"

namespace ocp
{

/// QP scaling (acados ocp_nlp_qpscaling.c).
///
/// Scales the staged QP in-place into `scaled_in_` and provides
/// `rescale_solution()` to map the solved (scaled-space) QpSol back to
/// original space.
///
/// Objective scaling: Gershgorin max-abs-eig of every stage Hessian and
/// the inf-norm of the gradient determine a single scalar factor.
/// Constraint scaling: each general-constraint row (ineq/eq/lin) gets a
/// per-row factor s_j = 1/max(1, max(bound_max, coeff_norm)); the
/// (u;x) Jacobian and d are multiplied by s_j, and for soft rows the
/// slack Hessian diagonal is divided by s_j^2 and the slack gradient by
/// s_j (acados-faithful, ocp_nlp_qpscaling.c:587-615).
///
/// Box and pin rows are NOT scaled (acados scales only `ng` rows).
template <class P, int NH = Eigen::Dynamic>
class QpScaler
{
public:
    using D = QpDim<P>;
    using S = typename P::scalar_t;

    struct Opts
    {
        bool scale_objective = false;
        bool scale_constraints = false;
        double ub_max_abs_eig = 1e5;
        double lb_norm_inf_grad_obj = 1e-4;
    };

    explicit QpScaler(Opts opts = {}) : opts_(opts)
    {
    }

    void resize(int n_stages)
    {
        scaled_in_.resize(n_stages);
        scaled_out_.resize(n_stages);
    }

    /// Scale the assembled (original-space) QP into scaled_in_.
    ///
    /// When neither scaling is active, scaled_in_ is a plain copy and
    /// obj_factor_ / min_constr_scaling_ stay 1.0.
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

    /// Map the solved (scaled-space) QpSol back to original space.
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

    /// Gershgorin-based objective scaling factor (acados
    /// ocp_nlp_qpscaling.c:483-540). Static so it can be unit-tested
    /// without a QpScaler instance.
    static double compute_obj_scaling_factor(const Qp<P, NH>& qp,
                                             double ub_max_abs_eig,
                                             double lb_norm_inf_grad_obj)
    {
        double max_abs_eig = 0.0;
        double grad_norm = 0.0;

        // Stage type 0 = first, 1..N-1 = path, N = terminal (acados loop
        // over stage 0..N). The (u;x) Hessian block plus the diagonal slack
        // weights Z: the max-abs-eig estimate is the Gershgorin bound over
        // the full (u;x;s) block, which equals max(ershgorin(H_ux),
        // max(Z)) because the slack block is diagonal and decoupled.
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

    /// Gershgorin max-abs-eigenvalue estimate (acados utils/math.c:1118).
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

    /// Scale every stage Hessian and gradient by `f`
    /// (acados ocp_qp_scale_objective, ocp_nlp_qpscaling.c:379-392).
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

    /// Per-row constraint scaling for one stage (acados
    /// ocp_nlp_qpscaling_scale_constraints, :545-618).
    /// Only general rows (ineq/eq/lin) are scaled; box/pin are untouched.
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

                // Soft-row slack cost rescale (acados :593-602):
                // Z_slack /= s^2, rqz_slack /= s, d_slack *= s (no-op
                // since d_slack = 0).
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

    /// Rescale the solved multipliers/slacks for one stage
    /// (acados rescale_solution_constraint_scaling, :327-375).
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

    Qp<P, NH> scaled_in_;
    QpSol<P, NH> scaled_out_;
    Opts opts_;
    double obj_factor_ = 1.0;
    double min_constr_scaling_ = 1.0;
    std::array<double, ngen_first> cs_first_{};
    std::array<double, ngen_path> cs_path_{};
    std::array<double, ngen_term> cs_term_{};
};

}  // namespace ocp
