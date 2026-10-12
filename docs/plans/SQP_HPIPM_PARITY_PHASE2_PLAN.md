# SQP + HPIPM parity — Phase 2: partial condensing (block grouping)

Status: **active** (`2a`, `2b`, `2c` complete). Builds on Phase 1 (full condensing,
`1a`–`1g` complete, committed). Adds acados' `PARTIAL_CONDENSING_HPIPM`: group the N stages into
N2 blocks of NB stages each, condense each block to a mini QP (the internal
states of a block are eliminated), and solve the resulting N2-stage "block"
QP with the *staged* HPIPM IPM. Mirrors acados
`ocp_qp_partial_condensing.c` + `x_part_cond.c`.

Reference files (read at implementation time; do not modify):
- `../acados/ocp_qp/ocp_qp_partial_condensing.c` (glue / dispatch)
- `../acados/external/hpipm/cond/x_part_cond.c`
  - `PART_COND_QP_COMPUTE_BLOCK_SIZE`, `PART_COND_QP_COMPUTE_DIM`
  - `PART_COND_QP_COND_LHS` (per-block `COND_BABT`, `COND_RSQ`, `COND_DCT`)
  - `PART_COND_QP_COND_RHS` (per-block `COND_B`, `COND_RQ`, `COND_D`)
  - `PART_COND_QP_COND_SOL` / `PART_COND_QP_EXPAND_SOL`
- `../acados/external/hpipm/cond/x_cond_aux.c` (the `COND_*` primitives)

## Key design (fixed up-front)

The condensed "block" QP is itself a **staged** OCP-QP with N2 block stages
plus a terminal copy stage (index N2). It is solved by the existing staged
`HpipmQpSolver` — no dense solver. Each block stage:

- variable vector `(u_blk; x_k0; s_blk)`:
  - `u_blk` = all NB control vectors of the block (nu_blk = NB·nu),
  - `x_k0` = the block's FIRST state only (internal states eliminated),
  - `s_blk` = all slack vars of the block's NB stages.
- dynamics = block flow map `x_{k0+NB} = BAbt_blk·[u_blk; x_k0] + b_blk`.
- internal state box rows (stages k0+1..k0+NB-1) become **general** rows
  (flow-mapped); the block-start state x_k0 box rows stay **box** rows.
- all ineq/eq/lin rows of the NB stages → general rows; all NB control box
  rows → box rows.

Block-stage dims (uniform; only block 0 differs by the pin-row group when
`fixed_initial_state`):
```
nbx_blk    = nbx                       (state box of x_k0; 0 if first & fixed)
nbu_blk    = NB * nbu
ngeneral   = NB*(ng + ne + nl) + (NB-1)*nbx
nslack_blk = nslack_path * NB          (block 0: nslack_first + (NB-1)*nslack_path)
nvar_blk   = NB*nu + nx + nslack_blk
nrow_blk   = (first? (fixed?nx:0) : 0) + nbx_blk + nbu_blk + ngeneral
```
Block 0 == "first" stage type; blocks 1..N2-1 == "path"; the terminal copy
== "term". This maps exactly onto the existing first/path/term typing.

Decisions:
- **Q1** `QpMode` gains `kPartialCondensed`. New `AcadosOptions.qp_cond_N`
  (int, 0 ⇒ N ⇒ identity fast path = solve staged directly; >0 ⇒ number of
  blocks N2, assert `N % (N/qp_cond_N) == 0` i.e. NB = N/N2 divides). The
  uniform block size NB is a **template** parameter (compile-time, decision
  Q3 of the master plan) so the block stage dims are compile-time.
- **Q2** `HpipmQpSolver` (and `Qp`/`QpSol`/`QpRes`/`QpStage*`) gain a
  `class Dim = QpDim<P>` template parameter; every structural dimension read
  goes through `Dim::`. `BlockQpDim<P, NB>` supplies the block-stage dims.
  All existing call sites keep compiling (default `Dim = QpDim<P>`).
- **Q3** `AcadosSolver` gains a trailing `int NB = 0` template param
  (0 = partial condensing disabled → `kPartialCondensed` falls back to
  staged with a printed note). NB>0 enables it. Appended last so the one
  existing 3-arg positional use (`AcadosSolver<DI, Dynamic, FailQp>`) still
  compiles — it will be updated to `AcadosSolver<DI, Dynamic, 0, FailQp>`.
- **Q4** `reduce_eq_dof` pre-pass (acados) is out of scope (no DAE `z`
  vars); pinned x0 stays as pin rows inside the block, exactly as in
  Phase 1.
- Warm start: the block QP is a staged QP, so dual warm-start (ws 2/3)
  persists in the block-solver workspace (`block_qp_out_` lam/t); primal
  cold, same convention as the other modes. `SqpIteration.cond_time`
  records block condense+expand wall clock (0 otherwise).

## Blocks

1. **2a — Dim-traits refactor. ✅** Added `class Dim = QpDim<P>` to
   `QpStageFirst/Path/Term`, `Qp`, `QpSol`, `QpRes`, `HpipmQpSolver`
   (default `QpDim<P>`, all existing call sites unchanged);
   `HpipmQpSolver` exposes `QpT/QpSolT/QpResT` aliases and routes all
   in-class references through them; `QpDim.fixed_initial_state` re-exposed
   so the solver's pin-row branch reads `D::` instead of `P::`. Note:
   `HpipmDenseQpSolver` is untouched (it only ever uses `DenseQp*` types).
   Gate: full warning-free build, all 22 targets byte-identical output to
   the pre-refactor baseline (modulo wall-clock cond_time lines).
2. **2b — `BlockQpDim<P, NB>` + dim tests. ✅** `BlockQpDim<P, NB>` added
   in `qp.hpp` (next to `QpDim`): mirrors the full `QpDim` member surface
   (nx, nu, nbx/nbu/ng/ne/nl, terminal dims, `fixed_initial_state`,
   nslack_*, lay_*, nvar_*/nrow_*/nside_*, side_*, idxb_*, idx_x0,
   nbound_*, idxs_lo/hi_*) with the block-stage values above. General rows
   are group-major (all ineq, then eq, then lin + internal state box); the
   block variable vector is `(u_blk; x_k0; s)` so `idx_x0` and the `idxb`
   state offsets shift by `NB*nu`; the `idxs` slack maps re-base each
   original per-stage slack column onto the block vector via a `remap`
   helper (preserves `-1` for non-soft rows). `NB = 1` reproduces
   `QpDim<P>` exactly. Test `tests/hpipm/block_qp_dim.cpp`: NB=1 identity
   vs `QpDim` (all members) for DI/MS/BlockRich(free)/BlockRichFixed;
   NB=2 extents/idxb/idxs + side mappings + slack-column bijection; and a
   faithful transcription of acados `PART_COND_QP_COMPUTE_DIM` cross-check
   (nbx2/nbu2/nb2/ng2/ns2) for N=4,8 NB=2. All 22 pre-existing targets stay
   byte-identical (2a/2b are additive; no behavior change).
3. **2c — `condense_block` (LHS+RHS). ✅**
   `solvers/hpipm/block_condense.hpp`: `condense_block<P,NH,NB>` fills a
   `Qp<P,Dynamic,BlockQpDim<P,NB>>` (N2 = N/NB). Per block: flow map
   `G[j]`/`gamma[j]` (x_{k0+j} = G[j]'(x_k0; u_prefix) + gamma[j]); block
   dynamics `BA = [B_blk | A_blk]` with `A_blk = G[NB]_{xx}'`,
   `B_j = G[NB]_{u_j}'`, `b = gamma[NB]`; H/g accumulated in h-order
   [x_k0; u_blk] (E_j'-substitution, Phase-1 style) then permuted to the
   block order (u_blk; x_k0; s_blk); slack diagonal + linear part per
   original stage; general rows (ineq/eq/lin of all stages + internal
   state boxes) flow-mapped with d-shift lo: `-dcx·gamma`, hi: `+dcx·gamma`
   (preserves the solver's residual form `d + t ∓ v`); state box j=0 and
   control boxes direct; pin rows (fixed x_0) direct; terminal copied
   verbatim. Test `tests/hpipm/condense_block.cpp`: (1) NB=1 bit-exact
   identity vs the staged QP; (2) NB>1 gates vs an independent forward
   simulation (BA, `F(z)-F(0) == ½z'Hz + g'z` — the QP objective drops the
   z-independent constant `F(0)`, DC rows + d/d_mask/m residual invariants,
   terminal copy); (3) end-to-end staged vs block solve: primal (u, x_k0,
   s, x_N) ≤1e-5, true OCP cost agreement, and N2=1 vs the Phase-1
   `condense_full` dense pipeline. All checks pass; 21 pre-existing
   targets unchanged (sqp_unit differs only in printed timing).
4. **2d — `expand_block_sol`. ✅**
   In `solvers/hpipm/block_condense.hpp` (inverse of `condense_block`):
   u/s sliced from the block stage vectors, x_k0 read from the block,
   internal states x_{k0+1}...x_{k0+NB-1} by forward simulation of the
   staged dynamics; lam/t copied row-group by row-group (inverse of the
   row-emit order, including the slack-side entries); pi_blk[ii] seeds
   the staged pi at k0+NB-1 (the composite-constraint multiplier carries
   the last stage of the dynamics chain) and the internal pi's come from
   the per-stage stationarity backward sweep (same sweep as
   `expand_full_sol`). Note: r_st must be the absolute row index in the
   staged stage layout (`lay.row_off(g) + local`), not the group-local
   index — a local index silently clobbers earlier groups (e.g. control
   box rows overwriting the pin rows of a fixed-x_0 first stage).
   Test `tests/hpipm/condense_block.cpp` Part 4: the expanded solution
   must (a) satisfy the staged QP KKT residuals (res_g ≤1e-5,
   res_b ≤1e-9, res_d ≤1e-7, res_m ≤1e-5 — verifies the dual mapping and
   the pi sweep without a second solve) and (b) match a direct staged
   solve on the primal (≤1e-5). For N2=1 it also matches the Phase-1
   `expand_full_sol` output: primal/pi ≤1e-5, and the duals compared as
   their unique parts only (per-row effective multiplier
   dm_hi·lam_hi − dm_lo·lam_lo and t on active sides ≤1e-4; individual
   two-sided lam entries are not unique for rows with both sides active,
   e.g. pins, and masked sides are unconstrained). All checks pass; the
   21 pre-existing targets are unchanged (sqp_unit differs only in
   printed timing).
5. **2e — driver hook.** `QpMode::kPartialCondensed`,
   `AcadosOptions.qp_cond_N`, `AcadosSolver<P, NH, NB>` member
   `block_qp_solver_` (`HpipmQpSolver<P, NH2, BlockQpDim<P,NB>>`) +
   `block_qp_in_/out_`; dispatch in `solve_qp` (main + SOC); identity fast
   path when `qp_cond_N >= N` or NB=1; `cond_time`; dual warm start via the
   block workspace. `options_2e` gains qp_cond_N default.
6. **2f — E2E + docs.** `tests/sqp/e2e_2f.cpp` (in `sqp_unit`):
   DoubleIntegrator + MassesChain in both `kStaged` and
   `kPartialCondensed` (NB=2, NB=4 where N%NB==0) from the same warm start
   agree ≤1e-8 (x/u inf, cost rel); iteration counts comparable.
   Update ARCHITECTURE/README + register `qp_cond_N` in the DUAL_ACCESS
   inventory. Mark this plan's blocks complete.

Gate for every block: `cmake --build build` warning-free
(`-Wall -Wextra -Werror`), full test suite green.

## Open items (verify during implementation, not design)

1. Whether `N2 = 1` (NB = N) is a useful degenerate check (it equals
   Phase 1's dense path in hard-block content) — use it as a cross-check,
   not a supported mode.
2. acados `PART_COND_QP_COND_LHS` uses the per-block `COND_*` primitives
   which themselves call the full-condense math on a bs-stage sub-QP;
   confirm our per-block accumulation matches (it should, by construction).
3. `cond_time` attribution when the identity fast path is taken (0, like
   staged).
4. Non-uniform `block_size` (HPIPM `bs0+1` for the first blocks) is a
   later generalization (master plan §9 item 4); uniform NB first.
