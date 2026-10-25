# Dual Access Plan (options & statistics by member or by name)

Advances TODO.md item 10; implements the "Separation of parameters" principle
in docs/ARCHITECTURE.md sec. 2 ("...access them as member variables of a
solver-specific options class or by a string name via a map"). It is the
per-object groundwork for TODO.md item 11 (central hierarchical discovery
map) and already implements the nested-group entry kind (`kRegistry` link +
dotted-path delegation) that item 11 will build on.

## Goal

Every run-time mutable scalar of the options and statistics structs must be
reachable in both ways, simultaneously, with zero migration cost:

1. direct member access, exactly as today (`opts.max_iter`, `stat.iter`, ...);
2. by name through a reference map (`opts.reg.set("max_iter", 40)`), making the
   values *discoverable* (enumerable, printable) and *settable* from an
   interactive environment. Nested values are reachable under dotted names
   that equal the C++ qualified member names
   (`opts.reg.get("glob.alpha_min", x)` == `opts.glob.alpha_min`).

The values stay where they are: plain, alloc-free struct/class members. The
registry only stores (name, offset, type) metadata and addresses the members
— it never copies or allocates a value.

## Current state (inventory)

| Object | Location | Contents |
| --- | --- | --- |
| `AcadosOptions` | `solvers/acados/acados.hpp:722` | 34 flat scalars (int/double/bool; `qp_cond_N` only effective with `qp_mode = QpMode::kPartialCondensed`) + `QpMode qp_mode` (enum) + nested `GlobOptions glob` |
| `GlobOptions` | `solvers/acados/globalize.hpp:38` | 5 scalars (embedded in `AcadosOptions`) |
| `RtiOptions` | `solvers/acados/rti.hpp:43` | 19 flat scalars (int/double/bool) + `QpMode qp_mode` (enum); RTI solver options (Phase 3) |
| `HpipmOptions` | `solvers/hpipm/hpipm.hpp:61` | 27 scalars (int/double/bool) |
| `AcadosStatistics` | `solvers/acados/acados.hpp:792` | `iter`, `status`, `rows` (10 columns x iter, `SqpIteration`) |
| `HpipmStatistics` | `solvers/hpipm/hpipm.hpp:129` | `iter`, `stat_max`, `status`, `rows` (21 columns, `HpipmIteration`) |

Usage constraints found in the code (the design must not break them):

- `AcadosOptions{}` / `HpipmOptions{}` as default arguments (acados.hpp:878,
  hpipm.hpp:355) and `HpipmOptions{}.tau_min`
  (tests/hpipm/centering_1g.cpp:275) -> an added user *default* constructor is
  fine, but aggregate-init with field values must not be relied on anywhere
  (it is not, verified by grep — including `GlobOptions`, which has no
  aggregate-init use).
- `AcadosOptions o2 = o;` copies (tests/sqp/driver_2g.cpp:110); solvers copy
  the options into a private `opts_` (acados.hpp:879, hpipm.hpp:355) -> the
  by-name map must remain valid on copies of the owning object (no stale
  pointers), including through the nested `GlobOptions` registry.
- Solvers mutate the QP solver's options through `set_warm_start` /
  `set_residual_tolerances` (hpipm.hpp:1837,1842) — these keep working
  untouched; the by-name surface is additive.

## Design

### 1. `ValueRegistry` (new header `include/ocp/named_values.hpp`)

Self-contained (no Eigen), ~200 lines. Core idea: **offset-based entries,
copy-safe by construction**, plus a `kRegistry` entry kind that links a child
`ValueRegistry` for hierarchical (dotted) names.

```cpp
namespace ocp
{

enum class ValueType : unsigned char { kDouble, kInt, kBool, kRegistry };

/// One named entry: short name + byte offset from the owner object start +
/// kind. For kRegistry the offset addresses a nested ValueRegistry member
/// (a hierarchical link, not a scalar).
struct NamedValue
{
    const char* name;      // short name at this level, no dot; string literal
    std::ptrdiff_t offset; // entry address - owner start
    ValueType type;
};

/// Read/write registry over the named scalars of one owning object.
///
/// The registry is a plain member of the owning struct (options /
/// statistics). All stored addresses are *layout-relative* (offsets from the
/// owner start), never raw pointers into the owner, so the implicitly
/// declared copy / move / assignment of the owner stays correct with no
/// per-copy re-registration: a copy has the identical layout, so the same
/// offsets are valid there. The registry must stay a direct (non-pointer)
/// member of the owner; by convention it is appended last in the struct.
class ValueRegistry
{
public:
    /// Record where the owner starts, relative to this member, and the
    /// owner size (range guard for add()). Call once, before the first add().
    template <class Owner>
    void begin(Owner& owner);

    /// Register a scalar under a short `name` (double / int / bool only;
    /// static_assert otherwise). Offsets are recorded layout-relative.
    /// Returns false if `value` does not live inside the owner.
    template <class T>
    bool add(const char* name, T& value);

    /// Register a child registry under a short `name` (hierarchical link).
    /// `child` must be a member of the owner (range-checked). Dotted lookups
    /// through this link are delegated to the child.
    bool add(const char* name, ValueRegistry& child);

    /// Type-safe read: false if `name` is missing or has a different type.
    /// Dotted names delegate to the linked child registry at the first dot.
    /// Never throws (project rule: no exceptions in the numerical core).
    template <class T>
    bool get(std::string_view name, T& out) const;

    /// Type-safe write through the map; const-qualified (mutates the
    /// pointee, not the map). Dotted names delegate as in get().
    /// Returns false on missing / type mismatch.
    template <class T>
    bool set(std::string_view name, T value) const;

    bool has(std::string_view name) const;   // delegates on dots
    std::size_t size() const;               // entries at this level
    /// Discovery: all leaf names at this level, children prefixed
    /// `<name>.` (flat, dotted, interactive environments).
    std::vector<std::string> names() const;
    /// Discovery: print flat dotted "name = value" lines (leaf scalars).
    void dump(std::FILE* f = stdout) const;

private:
    const NamedValue* find(std::string_view name) const;  // linear strcmp
    // type_of<double/int/bool>() helpers; ValueType::kDouble/kInt/kBool.

    std::ptrdiff_t owner_off_ = 0;   // owner start - this (member position)
    std::size_t owner_size_ = 0;     // sizeof(Owner), range guard for add()
    std::vector<NamedValue> entries_;
};

}  // namespace ocp
```

Implementation notes:

- `begin` stores `owner_off_ = (char*)&owner - (char*)this` and
  `owner_size_ = sizeof(Owner)`. `add` computes
  `offset = (char*)&entry - (char*)this - owner_off_` (for both scalar fields
  and child registries) and rejects it with `false` if
  `offset < 0 || offset >= owner_size_` (non-throwing precondition check: the
  entry must live inside the owning object, so a transient/heap
  `ValueRegistry` passed to `add(name, child)` is rejected). `get`/`set`
  resolve `addr = (char*)this + owner_off_ + offset` and
  `*reinterpret_cast<const T*>(addr)`. All offsets are layout constants; the
  copy of an owning struct is valid without any extra code.
- Delegation: `get`/`set`/`has` split the name at the first `.`. If the head
  segment resolves to a `kRegistry` entry, the child registry address is
  resolved and the remainder is resolved recursively in the child. Otherwise
  it is a scalar lookup at this level. `find` itself stays a short-name linear
  `strcmp`. Multi-level paths (`a.b.c`) fall out of the recursion; an empty
  head or a head that resolves to a scalar returns false.
- `names()` / `dump()`: recursive. `names()` returns the own short names and,
  for each registry link, the child's names prefixed `<head>.`; `dump()`
  prints flat dotted `name = value` lines for the leaf scalars (registry
  links themselves are not printed as values).
- Type safety via the `ValueType` tag + `static_assert` in `add`/`get`/`set`:
  `get<double>("max_iter")` on an int entry returns false, not UB; a type
  mismatch at any level of a dotted path returns false.
- Failure mode is `bool` (no exceptions, per AGENTS.md).
- Copy/move safety holds at every level of the hierarchy: a copy of the owner
  copies each nested registry at the identical offset, and every level's
  `owner_off_` and offsets stay valid — no per-copy re-registration.
- Per-level name uniqueness: short names (head segments) must be unique within
  one registry (a scalar member `glob` and a registry link `glob` cannot
  coexist). Links only ever point to nested members, so cycles are impossible.
- Linear `strcmp` lookup over `entries_` (a few dozen entries) — no hash map
  needed; keeps construction trivial and the header dependency-free.
- `entries_` is a `std::vector` built once at owner construction (metadata
  allocation, not in the numerical core). If a fully zero-allocation map is
  ever required, swap to a fixed `std::array<NamedValue, N>`; the API does not
  change. (TODO.md item 11's "alloc-free" requirement is about the *values*,
  which stay plain members.)

### 2. Wiring the options structs

Add `ValueRegistry reg;` as the last member and a default constructor that
registers every field (direct scalars, or the child registry for nested
structs). Direct member access is untouched; all existing call sites compile
unchanged.

```cpp
struct AcadosOptions
{
    // ... all existing fields, unchanged ...
    GlobOptions glob;

    ValueRegistry reg;                       // last member (convention)
    AcadosOptions()
    {
        reg.begin(*this);
        reg.add("max_iter", max_iter);
        reg.add("tol_stat", tol_stat);
        // ... one add() per flat field ...
        reg.add("glob", glob.reg);           // kRegistry link (nested)
    }
};
```

`GlobOptions` (globalize.hpp) gets its own registry:

```cpp
struct GlobOptions
{
    // ... existing 5 fields, unchanged ...

    ValueRegistry reg;                       // last member (convention)
    GlobOptions()
    {
        reg.begin(*this);
        reg.add("alpha_min", alpha_min);
        // ... one add() per field ...
    }
};
```

- Nested structs register *themselves*; the parent links them with
  `reg.add("glob", glob.reg)` instead of flat dot-prefixed scalar entries.
  The external name surface is identical to a flat dot prefix
  (`"glob.alpha_min"`), but ownership is local to `GlobOptions` — its naming
  surface travels with the struct if it is ever embedded elsewhere.
- `GlobOptions` gains a user-declared default constructor (no longer an
  aggregate); verified by grep that no aggregate-init of `GlobOptions` exists.
  The `glob_.opts = opts_.glob;` assignment (acados.hpp:1109) is unaffected
  (implicit copy-assignment of the registry copies the offset metadata, which
  is layout-identical and therefore valid).
- Same treatment for `HpipmOptions` (27 flat fields, no nesting).
- Copies: `AcadosOptions o2 = o;` is the implicit copy; `o2.reg.set(
  "glob.alpha_min", v)` correctly writes `o2.glob.alpha_min` and never
  touches `o` (offset-based design, section 1). This is the property the
  existing `AcadosOptions o2 = o;` A/B-comparison pattern relies on.

### 3. Wiring the statistics structs

Statistics have two kinds of values:

- **Scalars** (`iter`, `stat_max`, `status`): registered in a
  `ValueRegistry` exactly like options (no nesting in statistics, so no
  `kRegistry` links here). `status` (enum class `Status`, default underlying
  type `int`) is registered through its underlying integer:
  `reg.add("status", *reinterpret_cast<int*>(&status));` (legal lvalue
  reinterpret; `static_assert` that it is a scoped enum).
- **Per-iteration time series** (the `rows` columns: 10 for SQP, 21 for
  HPIPM): interactive access means "give me the whole column", not a pointer
  into one row. v1 keeps the existing row-major `std::vector<...Iteration>`
  storage (used by the solver main loop, `make_row`, `write_csv`, and ~15
  test files) and adds on-demand column extraction:

```cpp
struct AcadosStatistics
{
    int iter = 0;
    Status status = Status::kUnset;
    std::vector<SqpIteration> rows;
    // ... existing record() / write_csv() unchanged ...
    ValueRegistry reg;

    AcadosStatistics()
    {
        reg.begin(*this);
        reg.add("iter", iter);
        reg.add("status", *reinterpret_cast<int*>(&status));
    }
    /// Column names of one iteration row (discovery).
    static std::vector<std::string> column_names();
    /// Extract the named column over all recorded rows (false if unknown).
    bool column(std::string_view name, std::vector<double>& out) const;
};
```

- `column()` copies the column out of `rows` on demand (int columns cast to
  double). It is an interactive / post-hoc API, never in the solve hot path,
  so per-call allocation is acceptable.
- Deliberate non-goal for v1: reorganizing the statistics storage to
  column-major (which would let `column()` return a reference). Revisit only
  if column extraction ever becomes hot.

### 4. Discoverability surface (the interactive-environment aim)

For every options/statistics object:

- `reg.names()` — enumerable, dotted list of all settable/readable names
  (recursive over nested registries);
- `reg.dump()` — flat dotted `name = value` lines for a snapshot;
- `stat.column_names()` + `stat.column(name, out)` — the recorded series.

Usage in an interactive session:

```cpp
AcadosOptions opts;
for (const auto& n : opts.reg.names())
    std::cout << n << "  ";                 // discover, incl. glob.*
int mi = 0;
opts.reg.get("max_iter", mi);
opts.reg.set("max_iter", 40);              // same as opts.max_iter = 40
double am = 0.0;
opts.reg.get("glob.alpha_min", am);        // delegated to glob.reg
std::vector<double> rs;
solver.statistics().column("res_stat", rs); // iteration history for plotting
```

## Files

New:

- `include/ocp/named_values.hpp` — `ValueType` (incl. `kRegistry`),
  `NamedValue`, `ValueRegistry` (offset entries, dotted-path delegation,
  recursive discovery).
- `tests/registry/named_values.cpp` + registered in the root `CMakeLists.txt`
  as executable `named_values_unit` (pattern: the other test targets).

Modified:

- `include/ocp/solvers/hpipm/hpipm.hpp` — `HpipmOptions` (registry member +
  default constructor registering all 27 fields), `HpipmStatistics` (registry
  for `iter`/`stat_max`/`status`, `column_names()`, `column()`).
- `include/ocp/solvers/acados/acados.hpp` — `AcadosOptions` (registry +
  constructor: one `add()` per flat field + the `glob` `kRegistry` link),
  `AcadosStatistics` (registry for `iter`/`status`, `column_names()`,
  `column()`).
- `include/ocp/solvers/acados/globalize.hpp` — `GlobOptions` gains
  `ValueRegistry reg` as its last member plus a default constructor
  registering its 5 fields.
- `tests/hpipm/qp_unit.cpp`, `tests/sqp/options_2e.cpp` — extend with
  by-name round-trip checks (set via name -> read member; get via name ->
  compare; `names()` completeness incl. the `glob.*` leaf names; copy-safety:
  `AcadosOptions o2 = o;` then set via `o2.reg` — flat *and* dotted
  `glob.*` names — must not touch `o`).
- Root `CMakeLists.txt` — add the `named_values_unit` target.

## Execution steps

1. **Registry + unit tests.** Write `include/ocp/named_values.hpp` (incl.
   `kRegistry`, dotted delegation, recursive `names`/`dump`, range guard) and
   `tests/registry/named_values.cpp` (add/get/set/has/names/dump; type
   mismatch -> false; unknown name -> false; dotted delegation incl. a
   two-level chain; range guard rejects a transient registry; offset/copy
   semantics with a fixture struct incl. a nested struct field with its own
   registry; move + assignment). Gate: `cmake --build build` warning-free,
   `named_values_unit` passes. Checkpoint: registry done.
2. **HPIPM wiring.** Add the registry to `HpipmOptions` /
   `HpipmStatistics`; extend `tests/hpipm/qp_unit.cpp`. Gate: build +
   `qp_unit` + the example binaries run as before. Checkpoint: HPIPM done.
3. **Acados wiring.** Add the registry to `GlobOptions` (globalize.hpp) and
   link it from `AcadosOptions` (acados.hpp); add the registry to
   `AcadosStatistics`; extend `tests/sqp/options_2e.cpp` (dotted `glob.*`
   round-trip + nested copy-safety). Gate: build + `sqp_unit` + examples.
   Checkpoint: Acados done.
4. **Full regression.** Rebuild everything under `-Wall -Wextra -Werror`;
   run every test executable and example binary (double_integrator,
   mass_spring, masses_chain, sqp_masses_chain, masseschaingen_gen,
   sqp_masses_chain_gen_equiv, pendulum_on_cart, unicycle, sqp_pendulum,
   sqp_unicycle, sqp_p5probe_erk, sqp_p5probe_irk, qp_dim, qp_unit,
   sqp_unit, sqp_double_integrator, sqp_mass_spring, sqp_acados_ref,
   integrators_unit, sqp_continuous). Confirm zero behavior change (direct
   access is untouched; only new members/constructors were added).
   Checkpoint: item 10 complete.

## Non-goals (deferred to TODO.md item 11)

- The *solver-level root* registry on `AcadosSolver` / `HpipmQpSolver` that
  links the options, statistics, and sub-solver (QP solver, globalizer)
  registries into one hierarchy.
- Registering problem-side parameters and `Solution` fields.

The nested-group machinery item 11 anticipated (a "nested-group entry kind" on
`ValueRegistry` plus a path resolver) is already implemented here as the
`kRegistry` entry kind and dotted-path delegation. TODO item 11's
illustrative slash paths (`solver/hpipm/options/iter_max`) map 1:1 onto dotted
paths from a root registry (`hpipm.options.iter_max`); names equal the C++
qualified member names, so there is no separate resolution machinery to add.
Item 11 then reduces to: a root `ValueRegistry` per solver (a few
`add(name, child_reg)` calls) plus problem-side registration.

## Open decisions (resolved in this plan, recorded for review)

- **Map storage:** `std::vector<NamedValue>` (one allocation at owner
  construction, metadata only). Values themselves remain alloc-free members,
  as required by TODO.md item 11. Fixed-capacity `std::array` is the fallback
  if zero-allocation metadata becomes a hard requirement.
- **Nesting:** nested option structs own their own registry; the parent links
  it with `add(name, child.reg)` (`kRegistry` entry kind), and dotted lookup
  delegates recursively. This replaces the originally planned flat
  dot-prefixed registration of `glob.*` in `AcadosOptions`; the external name
  surface is unchanged (`glob.alpha_min` etc.) and equals the C++ qualified
  member names.
- **Names:** the C++ member names verbatim, snake_case (`max_iter`,
  `tol_stat`); nested structs are reached through registry links with the dot
  separator. No rename layer (the acados-style `i_max` in the TODO example
  was illustrative only).
- **Registration style:** manual per-field `reg.add()` calls in the owning
  default constructor. An X-macro single-source-of-truth (one list expanding
  to both the member declarations and the registrations) was considered and
  deferred: the light variant (macro generates registrations only, members
  stay explicit) is not single-source and adds preprocessor indirection; the
  full variant (macro also generates members) is awkward around the
  non-uniform entries (nested `glob` member, `status` enum, non-registered
  `rows`). Revisit for `HpipmOptions` (27 uniform fields) if item 11's
  discovery work makes the name-contract drift risk worth it.
- **Statistics series:** row-major storage kept, `column()` extracts on
  demand (copies). Column-major reorganization is a follow-up if it ever
  matters.
- **Enums:** registered through their underlying integer type
  (`reinterpret_cast<int&>`); documented at the call site.
- **QpSeed (Phase 4):** the QP sensitivity seeds
  (`solvers/hpipm/seed.hpp`) are plain data — no registry, no options
  struct, and the sensitivity API added no `HpipmOptions` fields (the
  KKT reg ladder reuses `reg_prim`).
