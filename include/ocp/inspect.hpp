// ocp/inspect.hpp
//
// inspect protocol: dual access (by member / by name) over the scalar
// fields of an object tree (docs/plans/OPTION_INSPECT_PLAN.md). Replaces
// the former offset-based ValueRegistry.
//
// Each type that exposes fields provides a static member
//
//     template <class Visitor>
//     static constexpr void inspect(Visitor&& vis);
//
// that calls `vis(name, &Self::member)` once per exposed field, where
//   - a scalar leaf is a double / int / bool member,
//   - a child is a member that itself provides inspect() (a nested
//     "registry" node reached via dotted names).
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

#pragma once

#include <cstddef>
#include <cstdio>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace ocp
{

namespace inspect
{

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
    template <class Name, class MemberPtr>
    void operator()(Name, MemberPtr) const
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

/// Walk `path` on `obj` and call `fn(leaf)` exactly once if a scalar
/// leaf is reached (first matching name per level); false if the path
/// is unknown or ends at a child.
template <class T, class Fn>
bool walk(T& obj, std::string_view path, const Fn& fn)
{
    const auto [head, rest] = split_first(path, '.');
    bool done = false;
    obj.inspect([&](std::string_view name, auto mem_ptr) {
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
                fn(m);
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
    T::inspect([&](std::string_view name, auto mem_ptr) {
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
    obj.inspect([&](std::string_view name, auto mem_ptr) {
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

}  // namespace detail

// ---------------------------------------------------------------
//  Field access (allocation-free)
// ---------------------------------------------------------------

/// Read the scalar field at dotted `path` from `obj` into `out`
/// (plain C++ assignment conversion). False if `path` is unknown or
/// ends at a child. Never throws.
template <class T, class Value>
bool get_field(const T& obj, std::string_view path, Value& out)
{
    return detail::walk(obj, path,
                        [&](const auto& leaf) { out = leaf; });
}

/// Write `value` into the scalar field at dotted `path` of `obj`
/// (plain C++ assignment conversion). False if `path` is unknown or
/// ends at a child.
template <class T, class Value>
bool set_field(T& obj, std::string_view path, Value&& value)
{
    return detail::walk(obj, path,
                        [&](auto& leaf) { leaf = std::forward<Value>(value); });
}

/// Double-coerced read (double-only hosts, e.g. MATLAB): int leaves
/// read as their numeric value, bool leaves as 1.0 / 0.0. False if
/// `path` is unknown or ends at a child.
template <class T>
bool get_double(const T& obj, std::string_view path, double& out)
{
    return detail::walk(obj, path, [&](const auto& leaf) {
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

/// Double-coerced write: the value is cast to the leaf's type (int via
/// static_cast, bool via `!= 0.0`). False if `path` is unknown or ends
/// at a child.
template <class T>
bool set_double(T& obj, std::string_view path, double value)
{
    return detail::walk(obj, path, [&](auto& leaf) {
        using M = std::decay_t<decltype(leaf)>;
        if constexpr (std::is_same_v<M, double>)
        {
            leaf = value;
        }
        else if constexpr (std::is_same_v<M, int>)
        {
            leaf = static_cast<int>(value);
        }
        else
        {
            leaf = value != 0.0;
        }
    });
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

/// The statistics column names (= the row struct's leaf field names).
/// Host/discovery use only (builds a vector of strings).
template <class Stats>
std::vector<std::string> stat_column_names(const Stats&)
{
    return leaf_names<row_t<Stats>>();
}

/// One named column: `out[i]` = rows[i]'s leaf, int/bool coerced to
/// double. False if the column name is unknown (also detects unknown
/// names on zero-row tables).
template <class Stats>
bool stat_column(const Stats& st, std::string_view name,
                 std::vector<double>& out)
{
    double v = 0.0;
    if (!get_double(row_t<Stats>{}, name, v))
    {
        return false;
    }
    out.resize(st.rows.size());
    for (std::size_t i = 0; i < st.rows.size(); ++i)
    {
        get_double(st.rows[i], name, out[i]);
    }
    return true;
}

/// All rows as a row-major double matrix (rows x cols, one column per
/// row-struct leaf, int/bool coerced to double). Host-side use only.
template <class Stats>
bool stat_table(const Stats& st, std::vector<double>& out, int& rows,
                int& cols)
{
    const std::vector<std::string> names = leaf_names<row_t<Stats>>();
    rows = static_cast<int>(st.rows.size());
    cols = static_cast<int>(names.size());
    out.assign(static_cast<std::size_t>(rows) *
                   static_cast<std::size_t>(cols),
               0.0);
    for (std::size_t i = 0; i < st.rows.size(); ++i)
    {
        for (std::size_t j = 0; j < names.size(); ++j)
        {
            get_double(st.rows[i], names[j],
                       out[i * names.size() + j]);
        }
    }
    return true;
}

}  // namespace inspect
}  // namespace ocp
