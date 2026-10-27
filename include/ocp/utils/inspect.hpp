// ocp/utils/inspect.hpp
//
// inspect protocol: dual access (by member / by name) over the scalar
// fields of an object tree (docs/plans/finished/OPTION_INSPECT_PLAN.md). Replaces
// the former offset-based ValueRegistry.
//
// Each type that exposes fields provides a static member
//
//     template <class Visitor>
//     static constexpr void inspect(Visitor&& vis);
//
// that calls `vis(name, &Self::member[, lo, hi])` once per exposed
// field, where
//   - a scalar leaf is a double / int / bool member,
//   - a child is a member that itself provides inspect() (a nested
//     "registry" node reached via dotted names),
//   - `lo` / `hi` (optional) are the closed-interval bounds enforced by
//     the named writers on that leaf: a number (int / double) or
//     std::nullopt for an unbounded side; kMinPositive is the
//     convenience bound for fields that must stay strictly positive.
// A class that may or may not expose fields inherits InspectableLeaf
// (no-op inspect) so the generic walkers below keep working.
//
// The walkers split dotted paths at the first dot and recurse into
// children. All traversal is allocation-free (string literals, member
// pointers, stack); the only allocating APIs are the host-side
// discovery / extraction helpers (leaf_names, stat_column_names,
// stat_column, stat_table), which are never used in the numerical path.
//
// Member pointers are formed inside the owning class's inspect(); C++
// does not re-check access when a member pointer is applied via .*,
// so private members can be exposed without making them public.
//
// Typed get_field / set_field convert like a plain C++ assignment
// (e.g. a double literal into an int leaf truncates); the double
// coerced get_double / set_double are the strict host (MATLAB) path.
// Names must be unique at each level; the first match wins.
//
// The named writers enforce the declared bounds on the double-coerced
// value before writing: NaN is always rejected, +/-inf is rejected
// against a finite bound (int / bool leaves reject any non-finite
// value), and an out-of-range value leaves the field untouched and
// yields kOutOfRange. Bounds are a host-side guardrail: the read
// path and direct member access bypass them by design.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace ocp
{

namespace inspect
{

/// Smallest positive normal double; the bounds constant for option
/// leaves that must stay strictly positive (denominators, barrier
/// floors).
constexpr double kMinPositive = std::numeric_limits<double>::min();

/// Base for a class that may or may not expose fields: the inherited
/// no-op inspect() lets the generic walkers recurse into it and find
/// zero leaves.
struct InspectableLeaf
{
    template <class Visitor>
    static constexpr void inspect(Visitor&&)
    {
    }
};

namespace detail
{

/// Visitor probe for the has_inspect trait (declared before the trait
/// below; a partial specialization's arguments are looked up at
/// definition point, so the probe must already be visible).
struct probe
{
    template <class Name, class MemberPtr, class... Limits>
    void operator()(Name, MemberPtr, Limits...) const
    {
    }
};

}  // namespace detail

/// True if `T` provides a static inspect(Visitor) (own or inherited);
/// C++17 replacement for a C++20 requires clause. Public so callers
/// (e.g. the MEX session) can SFINAE on "does this type expose
/// options" without naming any field.
template <class T, class = void>
struct has_inspect : std::false_type
{
};

template <class T>
struct has_inspect<T,
                   std::void_t<decltype(T::inspect(
                                   std::declval<detail::probe&>()))>>
    : std::true_type
{
};

namespace detail
{

template <class T>
constexpr bool is_scalar =
    std::is_same_v<std::decay_t<T>, double> ||
    std::is_same_v<std::decay_t<T>, int> ||
    std::is_same_v<std::decay_t<T>, bool>;

/// Split `path` at the first `delim`: {head, rest} (rest empty if none).
inline std::pair<std::string_view, std::string_view> split_first(
    std::string_view path, char delim)
{
    const auto pos = path.find(delim);
    if (pos == std::string_view::npos)
    {
        return {path, std::string_view{}};
    }
    return {path.substr(0, pos), path.substr(pos + 1)};
}

/// Interpret one limit operand: a number becomes the bound,
/// std::nullopt_t leaves that side unbounded.
template <class L>
constexpr std::optional<double> limit_of(L limit)
{
    if constexpr (std::is_same_v<std::decay_t<L>, std::nullopt_t>)
    {
        return std::nullopt;
    }
    else
    {
        return static_cast<double>(limit);
    }
}

/// Walk `path` on `obj` and call `fn(leaf, lo, hi)` exactly once if a
/// scalar leaf is reached (first matching name per level), passing the
/// leaf's declared bounds (std::nullopt: that side unbounded). False if
/// the path is unknown or ends at a child.
template <class T, class Fn>
bool walk(T& obj, std::string_view path, const Fn& fn)
{
    const auto [head, rest] = split_first(path, '.');
    bool done = false;
    obj.inspect([&](std::string_view name, auto mem_ptr, auto... limits) {
        if (done || name != head)
        {
            return;
        }
        auto& m = obj.*mem_ptr;
        using M = std::decay_t<decltype(m)>;
        if (rest.empty())
        {
            if constexpr (is_scalar<M>)
            {
                std::array<std::optional<double>, 2> bounds{
                    std::nullopt, std::nullopt};
                if constexpr (sizeof...(limits) == 2)
                {
                    std::size_t i = 0;
                    ((void)(bounds[i++] = limit_of(limits)), ...);
                }
                fn(m, bounds[0], bounds[1]);
                done = true;
            }
        }
        else if constexpr (has_inspect<M>::value)
        {
            done = walk(m, rest, fn);
        }
    });
    return done;
}

/// Collect the dotted leaf names reachable from `T::inspect`, depth
/// first in declaration order, under `prefix`.
template <class T>
void collect_names(std::string_view prefix, std::vector<std::string>& out)
{
    T::inspect([&](std::string_view name, auto mem_ptr, auto...) {
        using M = std::decay_t<decltype(std::declval<T>().*mem_ptr)>;
        if constexpr (is_scalar<M>)
        {
            out.push_back(std::string(prefix) + std::string(name));
        }
        else if constexpr (has_inspect<M>::value)
        {
            collect_names<M>(std::string(prefix) + std::string(name) + ".",
                             out);
        }
    });
}

/// Print flat dotted "name = value" lines (leaf scalars; children are
/// not printed as values themselves).
template <class T>
void dump_rec(const T& obj, const std::string& prefix, std::FILE* f)
{
    obj.inspect([&](std::string_view name, auto mem_ptr, auto...) {
        const auto& m = obj.*mem_ptr;
        using M = std::decay_t<decltype(m)>;
        if constexpr (is_scalar<M>)
        {
            if constexpr (std::is_same_v<M, double>)
            {
                std::fprintf(f, "%s%s = %.12e\n", prefix.c_str(),
                             name.data(), m);
            }
            else if constexpr (std::is_same_v<M, int>)
            {
                std::fprintf(f, "%s%s = %d\n", prefix.c_str(), name.data(), m);
            }
            else
            {
                std::fprintf(f, "%s%s = %s\n", prefix.c_str(), name.data(),
                             m ? "true" : "false");
            }
        }
        else if constexpr (has_inspect<M>::value)
        {
            dump_rec(m, prefix + std::string(name) + ".", f);
        }
    });
}


/// String counterpart of dump_rec: appends "name = value" lines to
/// `out` (identical formatting to dump()).
template <class T>
void dump_str_rec(const T& obj, const std::string& prefix, std::string& out)
{
    obj.inspect([&](std::string_view name, auto mem_ptr, auto...) {
        const auto& m = obj.*mem_ptr;
        using M = std::decay_t<decltype(m)>;
        if constexpr (is_scalar<M>)
        {
            char buf[64];
            if constexpr (std::is_same_v<M, double>)
            {
                std::snprintf(buf, sizeof(buf), "%.12e", m);
            }
            else if constexpr (std::is_same_v<M, int>)
            {
                std::snprintf(buf, sizeof(buf), "%d", m);
            }
            else
            {
                std::snprintf(buf, sizeof(buf), "%s", m ? "true" : "false");
            }
            out += prefix;
            out += name;
            out += " = ";
            out += buf;
            out += '\n';
        }
        else if constexpr (has_inspect<M>::value)
        {
            dump_str_rec(m, prefix + std::string(name) + ".", out);
        }
    });
}

}  // namespace detail

// ---------------------------------------------------------------
//  Field access (allocation-free)
// ---------------------------------------------------------------

/// Result of a named write (set_field / set_double).
struct SetResult
{
    enum class Status { kOk, kUnknownPath, kOutOfRange };

    Status status = Status::kUnknownPath;
    // The leaf's declared bounds; only meaningful when
    // status == kOutOfRange (+/- infinity: that side unbounded).
    double lo = 0.0;
    double hi = 0.0;

    bool ok() const { return status == Status::kOk; }

    /// Boolean context (legacy call sites): true on success.
    explicit operator bool() const { return ok(); }
};

/// Read the scalar field at dotted `path` from `obj` into `out`
/// (plain C++ assignment conversion). False if `path` is unknown or
/// ends at a child. Never throws.
template <class T, class Value>
bool get_field(const T& obj, std::string_view path, Value& out)
{
    return detail::walk(obj, path, [&](const auto& leaf, std::optional<double>,
                                       std::optional<double>) { out = leaf; });
}

/// Write `value` into the scalar field at dotted `path` of `obj`
/// (plain C++ assignment conversion), enforcing the field's declared
/// [lo, hi] bounds on the double-coerced value: NaN is always rejected,
/// +/-inf against a finite bound is rejected, and int / bool leaves
/// reject any non-finite value (they cannot represent it). An out-of-
/// range value leaves the field untouched. kOk on success,
/// kOutOfRange (with the bounds in lo / hi) on a violation,
/// kUnknownPath if `path` is unknown or ends at a child.
template <class T, class Value>
SetResult set_field(T& obj, std::string_view path, Value&& value)
{
    SetResult r;
    detail::walk(obj, path, [&](auto& leaf, std::optional<double> lo,
                                std::optional<double> hi) {
        using M = std::decay_t<decltype(leaf)>;
        const double v = static_cast<double>(value);
        const bool finite_ok =
            std::is_same_v<M, double> ? !std::isnan(v) : std::isfinite(v);
        if (!finite_ok || (lo.has_value() && v < lo.value()) ||
            (hi.has_value() && v > hi.value()))
        {
            r.status = SetResult::Status::kOutOfRange;
            r.lo = lo.value_or(-std::numeric_limits<double>::infinity());
            r.hi = hi.value_or(std::numeric_limits<double>::infinity());
            return;
        }
        leaf = std::forward<Value>(value);
        r.status = SetResult::Status::kOk;
    });
    return r;
}

/// Double-coerced read (double-only hosts, e.g. MATLAB): int leaves
/// read as their numeric value, bool leaves as 1.0 / 0.0. False if
/// `path` is unknown or ends at a child.
template <class T>
bool get_double(const T& obj, std::string_view path, double& out)
{
    return detail::walk(obj, path,
                        [&](const auto& leaf, std::optional<double>,
                            std::optional<double>) {
        using M = std::decay_t<decltype(leaf)>;
        if constexpr (std::is_same_v<M, double>)
        {
            out = leaf;
        }
        else if constexpr (std::is_same_v<M, int>)
        {
            out = static_cast<double>(leaf);
        }
        else
        {
            out = leaf ? 1.0 : 0.0;
        }
    });
}

/// Double-coerced write (strict host / MATLAB path): the value is
/// checked against the field's declared [lo, hi] bounds in the double
/// domain *before* any int/bool cast (an out-of-range fractional value
/// is rejected even if it would truncate into range), then converted
/// like a plain C++ assignment (int via static_cast, bool via
/// `!= 0.0`). Same SetResult semantics as set_field.
template <class T>
SetResult set_double(T& obj, std::string_view path, double value)
{
    return set_field(obj, path, value);
}

// ---------------------------------------------------------------
//  Discovery (host-side; the allocating APIs)
// ---------------------------------------------------------------

/// All leaf names reachable from `T::inspect`, dotted (child leaves
/// carry the "<child>." prefix), depth-first in declaration order.
/// Host/discovery use only (builds a vector of strings).
template <class T>
std::vector<std::string> leaf_names()
{
    std::vector<std::string> out;
    detail::collect_names<T>("", out);
    return out;
}

/// Print flat dotted "name = value" lines for every scalar leaf of
/// `obj` (host/discovery use only).
template <class T>
void dump(const T& obj, std::FILE* f = stdout)
{
    detail::dump_rec(obj, "", f);
}

/// All "name = value" lines of `obj` as a single string (same format
/// as dump()). Host/discovery use only (host-side allocation; the MEX
/// glue prints it via mexPrintf).
template <class T>
std::string dump_string(const T& obj)
{
    std::string out;
    detail::dump_str_rec(obj, "", out);
    return out;
}

// ---------------------------------------------------------------
//  Statistics tables (named columns)
// ---------------------------------------------------------------
//
// A statistics object exposes `int iter`, a status and
// `std::vector<Row> rows`; `Row` provides inspect() over its scalar
// fields. The row-struct field names are the column names (no
// per-value access, table access only).

/// Row type of a statistics object (the element type of `st.rows`).
template <class Stats>
using row_t = typename std::decay_t<
    decltype(std::declval<const Stats&>().rows)>::value_type;

/// Write one row's scalar leaves as consecutive doubles in declaration
/// order (the same order as leaf_names<Row>()), int/bool coerced to
/// double. `out` must hold leaf_names<Row>().size() doubles.
/// Allocation-free; host-side use (the stat table fills).
template <class Row>
void fill_row(const Row& row, double* out)
{
    std::size_t j = 0;
    row.inspect([&](std::string_view, auto mem_ptr, auto...) {
        const auto& m = row.*mem_ptr;
        using M = std::decay_t<decltype(m)>;
        if constexpr (detail::is_scalar<M>)
        {
            if constexpr (std::is_same_v<M, double>)
            {
                out[j] = m;
            }
            else if constexpr (std::is_same_v<M, int>)
            {
                out[j] = static_cast<double>(m);
            }
            else
            {
                out[j] = m ? 1.0 : 0.0;
            }
            ++j;
        }
        else if constexpr (has_inspect<M>::value)
        {
            fill_row(m, out + j);
        }
    });
}

/// The statistics column names (= the row struct's leaf field names).
/// Host/discovery use only (builds a vector of strings).
template <class Stats>
std::vector<std::string> stat_column_names(const Stats&)
{
    return leaf_names<row_t<Stats>>();
}

/// One named column: `out[i]` = rows[i]'s leaf, int/bool coerced to
/// double. `out` must hold st.rows.size() doubles. False if the column
/// name is unknown (also detects unknown names on zero-row tables).
template <class Stats>
bool stat_column(const Stats& st, std::string_view name, double* out)
{
    double v = 0.0;
    if (!get_double(row_t<Stats>{}, name, v))
    {
        return false;
    }
    for (std::size_t i = 0; i < st.rows.size(); ++i)
    {
        get_double(st.rows[i], name, out[i]);
    }
    return true;
}

/// All rows as a row-major double matrix (rows x cols, one column per
/// row-struct leaf, int/bool coerced to double). `out` must hold
/// rows*cols doubles with rows = st.rows.size() and cols = the row's
/// leaf count. Host-side use only.
template <class Stats>
bool stat_table(const Stats& st, double* out, int& rows, int& cols)
{
    const std::size_t ncol = leaf_names<row_t<Stats>>().size();
    rows = static_cast<int>(st.rows.size());
    cols = static_cast<int>(ncol);
    for (std::size_t r = 0; r < st.rows.size(); ++r)
    {
        fill_row(st.rows[r], out + r * ncol);
    }
    return true;
}

}  // namespace inspect
}  // namespace ocp
