// ocp/named_values.hpp
//
// ValueRegistry: a name -> (offset, type) registry over the scalar members
// of one owning object (e.g. an options or statistics struct). Entries are
// stored as byte offsets relative to the owner start, never as raw
// pointers into the owner, so the implicitly declared copy / move /
// assignment of the owner stays correct with no per-copy re-registration
// (a copy has the identical layout, hence the same offsets). The registry
// must be a direct (non-pointer) member of the owner and, by convention,
// its last member.

#pragma once

#include <cstddef>
#include <cstdio>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace ocp
{

/// Kind of a registry entry: a scalar (double / int / bool) or a link to a
/// nested ValueRegistry member (hierarchical, dotted names).
enum class ValueType : unsigned char
{
    kDouble,
    kInt,
    kBool,
    kRegistry
};

/// One named entry: short name (no dot, unique at this level) + byte offset
/// from the owner start + kind. For kRegistry the offset addresses a nested
/// ValueRegistry member (a hierarchical link, not a scalar).
struct NamedValue
{
    const char* name;
    std::ptrdiff_t offset;
    ValueType type;
};

/// Read/write registry over the named scalars of one owning object.
///
/// All stored addresses are *layout-relative* (offsets from the owner
/// start), never raw pointers into the owner. The registry is a plain
/// member of the owning struct; construction happens once in the owner's
/// default constructor (begin() followed by one add() per entry).
class ValueRegistry
{
public:
    /// Record where the owner starts, relative to this member, and the
    /// owner size (range guard for add()). Call once, before the first
    /// add(), from the owner's default constructor.
    template <class Owner>
    void begin(Owner& owner)
    {
        owner_off_ = (const char*)&owner - (const char*)this;
        owner_size_ = sizeof(Owner);
    }

    /// Register a scalar under a short `name` (double / int / bool only;
    /// static_assert otherwise). Offsets are recorded layout-relative.
    /// Returns false if `value` does not live inside the owner.
    template <class T>
    bool add(const char* name, T& value)
    {
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, int> ||
                      std::is_same_v<T, bool>,
                      "ValueRegistry::add supports double / int / bool only");
        const std::ptrdiff_t off =
            (const char*)&value - (const char*)this - owner_off_;
        if (off < 0 || (std::size_t)off >= owner_size_)
        {
            return false;
        }
        entries_.push_back(NamedValue{name, off, type_of<T>()});
        return true;
    }

    /// Register a child registry under a short `name` (hierarchical link).
    /// `child` must be a member of the owner (range-checked). Dotted
    /// lookups through this link are delegated to the child.
    bool add(const char* name, ValueRegistry& child)
    {
        const std::ptrdiff_t off =
            (const char*)&child - (const char*)this - owner_off_;
        if (off < 0 || (std::size_t)off >= owner_size_)
        {
            return false;
        }
        entries_.push_back(NamedValue{name, off, ValueType::kRegistry});
        return true;
    }

    /// Type-safe read: false if `name` is missing or has a different type.
    /// Dotted names delegate to the linked child registry at the first dot.
    /// Never throws (no exceptions in the numerical core).
    template <class T>
    bool get(std::string_view name, T& out) const
    {
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, int> ||
                      std::is_same_v<T, bool>,
                      "ValueRegistry::get supports double / int / bool only");
        const Resolved r = resolve(name, type_of<T>());
        if (r.entry == nullptr)
        {
            return false;
        }
        out = *reinterpret_cast<const T*>(r.scope->addr(r.entry->offset));
        return true;
    }

    /// Type-safe write through the map; const-qualified (mutates the
    /// pointee, not the map). Dotted names delegate as in get().
    /// Returns false on missing / type mismatch.
    template <class T>
    bool set(std::string_view name, T value) const
    {
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, int> ||
                      std::is_same_v<T, bool>,
                      "ValueRegistry::set supports double / int / bool only");
        const Resolved r = resolve(name, type_of<T>());
        if (r.entry == nullptr)
        {
            return false;
        }
        *reinterpret_cast<T*>(r.scope->addr(r.entry->offset)) = value;
        return true;
    }

    /// Double-coerced read for double-only host environments (MATLAB, ...):
    /// int entries read as their numeric value, bool entries as 1.0 / 0.0.
    /// Dotted names delegate as in get(). False if `name` is missing or is
    /// a registry link.
    bool get_as_double(std::string_view name, double& out) const
    {
        const Resolved r = resolve_any(name);
        if (r.entry == nullptr)
        {
            return false;
        }
        switch (r.entry->type)
        {
            case ValueType::kDouble:
                out = *reinterpret_cast<const double*>(
                    r.scope->addr(r.entry->offset));
                break;
            case ValueType::kInt:
                out = static_cast<double>(*reinterpret_cast<const int*>(
                    r.scope->addr(r.entry->offset)));
                break;
            case ValueType::kBool:
                out = *reinterpret_cast<const bool*>(
                          r.scope->addr(r.entry->offset))
                    ? 1.0
                    : 0.0;
                break;
            default:
                return false;
        }
        return true;
    }

    /// Double-coerced write: the value is cast to the entry's type (int via
    /// static_cast, bool via `!= 0.0`). Dotted names delegate as in set().
    /// False if `name` is missing or is a registry link.
    bool set_from_double(std::string_view name, double value) const
    {
        const Resolved r = resolve_any(name);
        if (r.entry == nullptr)
        {
            return false;
        }
        switch (r.entry->type)
        {
            case ValueType::kDouble:
                *reinterpret_cast<double*>(r.scope->addr(r.entry->offset)) = value;
                break;
            case ValueType::kInt:
                *reinterpret_cast<int*>(r.scope->addr(r.entry->offset)) =
                    static_cast<int>(value);
                break;
            case ValueType::kBool:
                *reinterpret_cast<bool*>(r.scope->addr(r.entry->offset)) =
                    value != 0.0;
                break;
            default:
                return false;
        }
        return true;
    }

    /// True if `name` resolves to an entry of any kind; delegates on dots.
    bool has(std::string_view name) const
    {
        const std::size_t dot = name.find('.');
        const NamedValue* head =
            find(dot == std::string_view::npos ? name : name.substr(0, dot));
        if (head == nullptr)
        {
            return false;
        }
        if (dot == std::string_view::npos)
        {
            return true;
        }
        if (head->type != ValueType::kRegistry)
        {
            return false;
        }
        const ValueRegistry* child =
            reinterpret_cast<const ValueRegistry*>(addr(head->offset));
        return child->has(name.substr(dot + 1));
    }

    /// Number of entries registered at this level.
    std::size_t size() const
    {
        return entries_.size();
    }

    /// Discovery: all leaf names at this level, children prefixed
    /// `<name>.` (flat, dotted, for interactive environments).
    std::vector<std::string> names() const
    {
        std::vector<std::string> out;
        for (const NamedValue& e : entries_)
        {
            if (e.type == ValueType::kRegistry)
            {
                const ValueRegistry* child =
                    reinterpret_cast<const ValueRegistry*>(addr(e.offset));
                const std::string prefix = std::string(e.name) + ".";
                for (const std::string& child_name : child->names())
                {
                    out.push_back(prefix + child_name);
                }
            }
            else
            {
                out.push_back(e.name);
            }
        }
        return out;
    }

    /// Discovery: print flat dotted "name = value" lines (leaf scalars;
    /// registry links themselves are not printed as values).
    void dump(std::FILE* f = stdout) const
    {
        dump_rec("", f);
    }

private:
    /// A resolved entry plus the registry that owns it (the offsets of
    /// nested entries are relative to that registry's owner, not the
    /// registry the lookup started from).
    struct Resolved
    {
        const NamedValue* entry;
        const ValueRegistry* scope;
    };

    /// Resolve `name` (possibly dotted) to a scalar entry of `expected`
    /// type, or to a registry entry if `expected` is kRegistry. Dotted
    /// names split at the first dot and delegate to the linked child.
    Resolved resolve(std::string_view name, ValueType expected) const
    {
        const std::size_t dot = name.find('.');
        if (dot != std::string_view::npos)
        {
            if (expected != ValueType::kRegistry)
            {
                // A dotted path always ends at a scalar in the child; the
                // head must be a registry link.
                const NamedValue* head = find(name.substr(0, dot));
                if (head == nullptr || head->type != ValueType::kRegistry)
                {
                    return {nullptr, this};
                }
                const ValueRegistry* child =
                    reinterpret_cast<const ValueRegistry*>(addr(head->offset));
                return child->resolve(name.substr(dot + 1), expected);
            }
            return {nullptr, this};  // kRegistry never resolves through a dot
        }
        const NamedValue* entry = find(name);
        if (entry == nullptr || entry->type != expected)
        {
            return {nullptr, this};
        }
        return {entry, this};
    }

    /// Resolve `name` (possibly dotted) to a scalar entry of any kind
    /// (kDouble / kInt / kBool); registry links are rejected.
    Resolved resolve_any(std::string_view name) const
    {
        const Resolved r = resolve(name, ValueType::kDouble);
        if (r.entry != nullptr)
        {
            return r;
        }
        const Resolved r2 = resolve(name, ValueType::kInt);
        if (r2.entry != nullptr)
        {
            return r2;
        }
        return resolve(name, ValueType::kBool);
    }

    const NamedValue* find(std::string_view name) const
    {
        for (const NamedValue& e : entries_)
        {
            if (std::string_view(e.name) == name)
            {
                return &e;
            }
        }
        return nullptr;
    }

    char* addr(std::ptrdiff_t offset) const
    {
        return const_cast<char*>((const char*)this + owner_off_ + offset);
    }

    void dump_rec(const std::string& prefix, std::FILE* f) const
    {
        for (const NamedValue& e : entries_)
        {
            if (e.type == ValueType::kRegistry)
            {
                const ValueRegistry* child =
                    reinterpret_cast<const ValueRegistry*>(addr(e.offset));
                child->dump_rec(prefix + e.name + ".", f);
            }
            else if (e.type == ValueType::kDouble)
            {
                std::fprintf(f, "%s%s = %.12e\n", prefix.c_str(), e.name,
                             *reinterpret_cast<const double*>(addr(e.offset)));
            }
            else if (e.type == ValueType::kInt)
            {
                std::fprintf(f, "%s%s = %d\n", prefix.c_str(), e.name,
                             *reinterpret_cast<const int*>(addr(e.offset)));
            }
            else
            {
                std::fprintf(f, "%s%s = %s\n", prefix.c_str(), e.name,
                             *reinterpret_cast<const bool*>(addr(e.offset))
                                 ? "true"
                                 : "false");
            }
        }
    }

    template <class T>
    static constexpr ValueType type_of()
    {
        if constexpr (std::is_same_v<T, double>)
        {
            return ValueType::kDouble;
        }
        else if constexpr (std::is_same_v<T, int>)
        {
            return ValueType::kInt;
        }
        else
        {
            return ValueType::kBool;
        }
    }

    std::ptrdiff_t owner_off_ = 0;   // owner start - this (member position)
    std::size_t owner_size_ = 0;     // sizeof(Owner), range guard for add()
    std::vector<NamedValue> entries_;
};

}  // namespace ocp
