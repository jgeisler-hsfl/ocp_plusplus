# Statistics ring buffer: compile-time fixed storage, per-row iteration number

## Goal
Replace the heap-backed `std::vector<Row>` statistics tables with a
compile-time fixed-size **FIFO ring buffer**, and give every statistics row
an explicit **iteration number** (`int iter`) field so the exported table is
self-describing even after wrap-around.

## Design decisions (agreed)
- **FIFO (drop oldest)**, not "drop beyond size". Failure diagnostics live in
  the *tail*; typical runs (SQP <= 20, IPM <= 50) fit in the capacity so the
  two policies only differ on pathological long runs — exactly the ones to
  debug.
- Ring logic is **encapsulated** in a generic `ocp::RingBuffer<T, N>` class
  (new header `include/ocp/utils/ring_buffer.hpp`). Solver / statistics code
  never sees a modulo or slot index.
- No `std::` FIFO is suitable: `std::queue` is unbounded (heap `deque`),
  offers no indexed read access, and cannot expose a whole-table view for
  CSV / MEX / inspect. A ~50-line own header keeps the Eigen-only constraint.
- Capacity default `NSlots = 64` (>= the `iter_max = 50` default, so nothing
  is dropped in normal operation).
- `HpipmOptions::stat_max` is **removed** (the compile-time `NSlots` replaces
  its role; a `iter_max > 64` now FIFO-drops the oldest instead of truncating
  at 50).

## Core: `ocp::RingBuffer<T, N>` (`utils/ring_buffer.hpp`)
```cpp
template <class T, int N>  // N >= 1
class RingBuffer {
public:
    using value_type = T;
    static constexpr int kCapacity = N;
    void reset();                  // drop all entries
    bool empty() const;
    bool full() const;
    int  size() const;            // valid entries = min(total writes, N)
    T&   next();                  // next write slot; evicts oldest when full
    const T& at(int i) const;     // i-th valid entry, oldest (0) -> newest
    const T& newest() const;      // most recent (read)
    T&       newest();            // most recent (write)
private:
    std::array<T, N> data_{};
    int head_ = 0;   // next write slot (oldest entry when full)
    int size_ = 0;   // valid entries
};
```
- `next()`: write at `head_`, `++size_` (while < N), advance `head_ = (head_+1) % N`.
- `at(i)`: `data_[(head_ - size_ + i + N) % N]` (always non-negative since `size_ <= N`).
- No allocation, no per-write zeroing (the caller fills `next()`'s slot).

## Row structs gain `int iter = 0;` as first field + first inspect leaf
- `SqpIteration` (acados.hpp), `HpipmIteration` (hpipm.hpp), `RtiIteration`
  (rti.hpp).
- This makes the inspect column count match the CSV/MEX column count (the
  "iter" first column becomes a real leaf) and the table self-describing.

## `AcadosStatistics<NSlots = 64>` (acados.hpp)
```cpp
template <int NSlots = 64>
struct AcadosStatistics {
    using Row = SqpIteration;
    int iter = 0;                 // number of recorded rows
    Status status = Status::kUnset;
    utils::RingBuffer<Row, NSlots> rows_;
    int n_rows() const { return rows_.size(); }
    const Row& row(int i) const { return rows_.at(i); }
    void record(const Row& in) {
        Row& slot = rows_.next();
        slot = in;
        slot.iter = iter;          // 0-based SQP loop index
        ++iter;
    }
    void write_csv(const char* path) const;  // label from row.iter
};
```
- `AcadosSolver::solve()` already does `stat_ = AcadosStatistics{};` (no
  change). `make_row()` unchanged. `record()` call sites unchanged.
- `stat_` member becomes `AcadosStatistics<> stat_;`; `statistics()` returns
  `const AcadosStatistics<>&`.

## `HpipmStatistics<NSlots = 64>` (hpipm.hpp)
Row 0 is the pre-iteration snapshot and is kept **outside** the ring (it is the
initial-state snapshot and is never overwritten); the ring holds post-iteration
rows only.
```cpp
template <int NSlots = 64>
struct HpipmStatistics {
    using Row = HpipmIteration;
    int iter = 0;                 // IPM iterations performed
    Status status = Status::kUnset;
    Row pre{};                    // pre-iteration residuals (label 0)
    utils::RingBuffer<Row, NSlots> ring_;
    int n_rows() const { return 1 + ring_.size(); }
    const Row& row(int i) const { return (i == 0) ? pre : ring_.at(i - 1); }
    Row& pre_row() { return pre; }               // initial / final fill
    Row& next_row() {                            // one post-iteration row
        Row& slot = ring_.next();
        slot = Row{};
        ++iter;
        slot.iter = iter;                         // 1-based IPM iteration
        return slot;
    }
    Row& last_row() { return ring_.newest(); }    // final-residuals fill
    void reset() { iter = 0; status = Status::kUnset; pre = Row{}; ring_.reset(); }
    void print() const;                            // n_rows rows, label row.iter
    void write_csv(const char* path) const;
};
```

## HPIPM solve-loop rewrites (staged `solve`, `solve_abs`, dense `solve`, both
`solve_unconstr`)
- `stat_.init(opts_.stat_max)` -> `stat_.reset()`.
- `fill_stat_residuals(stat_.row(0), ...)` -> `fill_stat_residuals(stat_.pre_row(), ...)`.
- `HpipmIteration* row = stat_row(kk);` + all `if (row) { row->... }` ->
  `Row& row = stat_.next_row();` + direct `row....` (guards disappear).
- Remove now-redundant `stat_.iter = kk + 1;` (stamped by `next_row()`) and the
  `stat_.iter = kk;` / `stat_.iter = 0;` in the failure / unconstr paths.
- Final-residuals fill `fill_stat_residuals(stat_.row(stat_.iter), ws_.res)`
  (guarded by `iter < stat_max`) -> `if (stat_.iter > 0) fill_stat_residuals(stat_.last_row(), ws_.res);`.
- Delete the `stat_row(int)` helpers (staged + dense).
- `stat_` members become `HpipmStatistics<> stat_;`; `statistics()` returns
  `const HpipmStatistics<>&`.

## `RtiIteration` / `RtiStatistics` (rti.hpp)
- `RtiIteration` gains `int iter = 0;` (first field).
- `record_row()` sets `r.iter = stat_.n_calls;` (0-based call index) before
  `stat_.n_calls += 1;`. RTI keeps its single `last` row (out of scope for the
  ring).

## inspect protocol (`utils/inspect.hpp`)
- `row_t<Stats>` = `typename Stats::Row`.
- `stat_column` / `stat_table` iterate `st.n_rows()` and read `st.row(i)`.
- Header comment updated from `std::vector<Row> rows` to the
  `n_rows()` / `row(i)` / `using Row` contract.

## MEX (`include/ocp/mex/mex_api.hpp`, `mex/ocp_mex_impl.hpp`)
- Solver contract comment: `std::vector<Row> rows` -> `n_rows()`,
  `const Row& row(int)`, `using Row`.
- `stat_fill` uses `st.row(r)` instead of `st.rows[r]`.
- `do_stats`: row count stays `stat_iter()` (== `n_rows()` for SQP); the struct
  array now carries an `iter` field (row labels survive wrap-around).

## Tests
- **New** `tests/utils/ring_buffer.cpp`: fill-to-capacity, wrap-around order,
  `at()`/`newest()`/`size()` after overwrites, `reset()`; plus a statistics
  overflow test (record > `NSlots` -> oldest dropped, `iter` fields correct,
  order preserved).
- `qp_unit.cpp`: drop `stat_max` checks; `check_stat_table` uses `next_row()` /
  `n_rows()`; column count 21 -> 22 (leading "iter").
- `solve_1h.cpp`: `stat.rows.size() == stat_max + 1` -> `stat.n_rows() == iter + 1`;
  `stat.row(it)` stays.
- `suite_1i.cpp`: drop `opts.stat_max = 2;`.
- `options_2e.cpp`: `AcadosStatistics` record test uses `n_rows()` / `row(i)`.
- `stats_5d.cpp`: CSV expectations unchanged (labels identical, no overflow).
- `funnel_3i.cpp`, `rh_shift.cpp`, `warm_start_3c.cpp`, `driver_2g.cpp`,
  `qpmode_1e.cpp`, `rti_3d.cpp`, `abs_form_3e.cpp`,
  `sqp_p5probe_{erk,irk}.cpp`: range-for / `.rows` / `.rows.back()` / `.rows[0]`
  -> `n_rows()` / `row(i)` / `row(n_rows()-1)`.
- `mex_api_unit.cpp`: verify `stat_fill` / `stat_column` against the new
  accessors.

## Build
- Add `ring_buffer_unit` (or fold into an existing target) in
  `CMakeLists.txt`.

## Docs
- `docs/ARCHITECTURE.md`, `README.md`, `docs/tutorial/` where the statistics
  tables / `stat_max` are described.
- Mark the corresponding `docs/plans/TODO.md` item done before committing.

## Execution blocks
1. [x] Recon: enumerate every statistics consumer (done).
2. [x] `utils/ring_buffer.hpp` + unit test.
3. [x] Row structs: add `iter` field + inspect leaf.
4. [x] `AcadosStatistics<>` (ring + record + csv).
5. [x] `HpipmStatistics<>` (ring + pre row + accessors + csv/print).
6. [x] HPIPM solve loops (staged / abs / dense / unconstr) + drop `stat_max`.
7. [x] RTI `iter` field.
8. [x] inspect protocol (`row_t`, `stat_column`, `stat_table`).
9. [x] MEX session contract + `stat_fill`; `do_stats` (via `stat_nrows()`).
10. [x] Update all affected tests + new ring/overflow test; CMake wiring.
11. [x] Docs (ARCHITECTURE, README, tutorial, TODO).

## Findings
- `HpipmStatistics::last_row()` needed both const and non-const overloads:
  the final-residuals fill in `solve_abs` writes through it, while
  `stats_5d.cpp` reads `stat.last_row().mu` on a `const` statistics object.
- MEX `do_stats` / `do_stat_col` are driven by the *stored* row count
  (`stat_nrows()`), not the total recorded count (`stat_iter()`); the
  per-row `iter` field is what makes the table self-describing after the
  ring wraps.
- HPIPM stats table is now 22 columns (leading `iter`); SQP 12.
- Test fixes beyond the API rename: `qp_unit.cpp` copy-safety test switched
  from `stat_max` to `itref_pred_max` (must not collide with the earlier
  `iter_max` set); `mex_api_unit.cpp` option count 66 -> 65, stat columns
  11 -> 12 (leading `iter`).
