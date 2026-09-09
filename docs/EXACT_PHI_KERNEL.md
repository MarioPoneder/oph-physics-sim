# Exact potential-linked stochastic repair kernel (lane L5)

Module `oph_exact/phi_kernel.py`, receipt `data/exact/phi_linked_kernel_receipt.json`
(schema `oph.exact.phi-linked-kernel.v1`), independent verifier
`oph_exact/verify_phi_kernel_independent.py`, tests `tests/test_exact_phi_kernel.py`.

## 1. The declared kernel

State space. Nonnegative integer loads on the twelve ports of the icosahedral
carrier with a protected total `s`, enumerated as weak compositions: 364
states at `s = 3`, 1365 states at `s = 4`.

Moves. The sixty directed seam labels `(c, o)`, two per seam. Label `(c, o)`
waits when the seam gap `|x_c - x_o|` is at most one and otherwise sends the
endpoint pair to `(ceil(t/2), floor(t/2))` with the ceiling at `c`, where `t`
is the endpoint total. On an even gap both labels of a seam give the same
state; on an odd gap of at least three they give two distinct states. The
move law is shared by both potentials below; only the tilt differs.

Potential axis. Two potentials are declared.

- `squared_norm_V` (primary): `V(x) = sum_i x_i^2`, the flagship's quadratic
  descent functional. A unit transfer across a seam with oriented mismatch
  `d >= 2` lowers `V` by exactly `2 (d - 1)`. A declared move with seam gap
  `D` transfers `floor(D/2)` or `ceil(D/2)` units and lowers `V` by
  `2 floor(D/2) ceil(D/2)`, the sum of the unit-transfer decrements
  `2 (D - 2k - 1)`, for either placement.
- `seam_sum_Phi` (secondary, diagnostic): `Phi(x) = sum over the thirty seams
  of (x_i - x_j)^2`, the carrier mismatch potential of `oph_exact/carrier.py`
  on integer states. Its terminal value varies over the agreement shell.

Kernel family. One label per move,

    P_beta(m | x) = 2^(-beta dU_m(x)) / Z_beta(x),   dU_m(x) = U(move(x, m)) - U(x),

for the declared potential `U` and `beta` in `{0, 1, 2}`. `beta = 0` is the
uniform A3 schedule (`1/60` per directed label), identical for both
potentials. Every decrement is nonpositive, so every weight is a positive
integer power of two and every row normalizer is a positive integer; every
probability is an exact rational.

### The gap-one reading

The bounded self-readback closure reads the reverse label on a gap-one seam
as a reversible endpoint swap, a neutral motion of `V`. The receipt records
the literal reading on every state-label pair: at `s = 3` it differs from the
declared move on 3360 pairs, every one of them `V`-neutral, of which 900 raise
`Phi` (maximum `+4`, witness state `(2, 1, 0, ...)`, label `(8, 1)`), 1560
leave `Phi` unchanged and 900 lower it; at `s = 4` the counts are 13800 (all
`V`-neutral), 4320 (maximum `+6`), 5160 and 4320. A strict-descent law
therefore waits on gap-one seams, which is the declared move law. Under it
every state-changing move lowers both `V` and `Phi` in both sectors, the
decrement is zero exactly on waits, and the receipt carries the exhaustive
check over all 21840 and 81900 state-label pairs (minimum decrements `-4` and
`-8` for `V`, `-24` and `-48` for `Phi`).

## 2. The corpus statements the receipt inhabits

Flagship, `reverse-engineering-reality/flagship/from_observer_consensus_to_standard_physics.tex`,
lines 889 to 895, the quadratic descent functional:

> record keeping is integer valued: a write appends +1, a retraction appends
> -1, and readback sums atomic events per port; a conservative repair
> transfers one unit across a seam whenever the oriented mismatch has
> magnitude at least two. With V(N) = sum_i N_i^2, each such repair strictly
> decreases V by 2(d-1) for mismatch d >= 2, so repair terminates by a finite
> theorem.

The consensus normal-form theorem's exact quadratic descent is this `V`. The
receipt verifies the identity `dV = -2 (d - 1)` on every unit transfer of both
sectors (660 and 4020 state-seam pairs), the closed form
`dV = -2 floor(D/2) ceil(D/2)` on every declared move, and the equality of
the declared-move decrement with the sum of its unit-transfer decrements
(1260 unit transfers and 60 two-unit moves at `s = 3`; 7320 and 720 at
`s = 4`).

Instantiation paper, subsection "(M5) A source-selected action"
(`trt-scspl/instantiating_the_self_configuring_self_processing_language_that_is_our_universe.tex`,
lines 1665 to 1715 at the time of writing, referenced without a hash):

> The corpus has a derived action, fixed by the realized kernel and the
> step-uniform reference, with Hamilton equations; it is not source-selected
> because the source corner table does not identify the real Legendre
> continuation. What would connect Phi to an action is a declared stochastic
> repair law whose log-transition action is a function of the Phi decrement
> along each accepted move. With that law the derived-action theorem applies
> and the source selects the action up to the non-identifiability family. The
> flagship names a separately source-justified stochastic coupling as a
> missing premise of its conditional four-law theorem, where the
> strict-descent normalizer carries no entropy inequality; a Phi-linked action
> needs the same object. No such law is declared.

The instantiation paper's "Phi decrement" is read here as the flagship's `V`
decrement: the flagship's exact quadratic descent is `V`, and under `V` the
agreement shell is the `V`-minimal set of the sector. The seam-sum `Phi` of
the carrier module is kept as a diagnostic variant; its terminal value is
schedule dependent. The receipt declares one such law and verifies the
consequences the passage lists. Declaring the law does not select it from the
source; (M5) stays open.

Lean, `reverse-engineering-reality/Lean/InformationProjection/LogTransitionAction.lean`,
module header: for a strictly positive row-stochastic kernel and an initial
law, the Markov path law equals the exponential tilt of the declared reference
(initial law times the uniform-step counting weight `card^(-n)`) by the
log-transition action `S(gamma) = -sum log P` at multiplier one, with
partition constant `card^(-n)` (`markov_path_law_eq_gibbs`,
`markov_tiltZ_eq`); two action-multiplier pairs produce the same tilt at a
fixed positive reference exactly when their multiplier-weighted actions differ
by one additive constant (`tilt_eq_tilt_iff_gauge`,
`action_unique_up_to_gauge`); the bare log action forces multiplier one only
when the action is nonconstant
(`bare_log_action_multiplier_unique_of_nonconstant`).

The receipt instantiates this on the label path space: from a fixed initial
state the label kernel is strictly positive (all sixty labels carry positive
weight at every state), the reference is `pi(x_0) 60^(-n)`, and the path law
is `pi(x_0) prod_k P(m_k | x_{k-1})`. The state-space kernel has zero entries
and sits outside the theorem's positivity hypothesis; the label lift is the
strictly positive object.

Observers paper, `reverse-engineering-reality/paper/observers_are_all_you_need.tex`,
lines 3076 to 3103, the definition of `L_0` and `y`:

> Let `L_0(x,y)` be the bilinear real extension of the committed two-state
> log-transition table. It is affine in `y`, so its momentum at fixed `x` has
> singleton image and admits no global velocity solver. For every `a` in `R`,
> define `L_a(x,y) = L_0(x,y) + (a/2) y (y-1)`. Every `L_a` agrees with the
> exact source log-transition action on every realized binary history at every
> path length. [...] In particular the checked `a=1` and `a=2` members have
> distinct Lagrangians and Hamiltonians, agree on all source histories but
> differ by `1/8` when one middle record is varied to `y=1/2`.
>
> The correction `y(y-1)` vanishes at both source symbols.

So `y` is the target record slot of a two-point Lagrangian on the binary
alphabet, valued in `{0, 1}` on realized histories, and `L_0` is the
log-transition table extended affinely in that slot. The flagship states the
same theorem in its boundary at lines 3006 to 3021: "every
`L_a = L_0 + (a/2) y (y-1)` with `a > 0` agrees on every realized history yet
is strictly convex with an explicit Hamiltonian".

Flagship four-law premise, table row at lines 432 to 436 and the boundary at
lines 4308 to 4319:

> The arrow of time: every finite stochastic repair kernel preserving a
> supplied faithful reference contracts relative entropy; the deterministic
> strict-descent normalizer need not, as an exact counterexample shows.
>
> The strict-descent normalizer that settles public facts is a different map
> and carries no entropy inequality, with an explicit certified two-point
> counterexample. [...] Physical thermodynamics additionally requires
> source-justified transitions, a shared reference, energy-clock calibration
> and uniform low-temperature tails on one cofinal family.

Lean, `reverse-engineering-reality/Lean/Thermodynamics/CommonReferenceObstruction.lean`,
lines 22 to 26:

> These statements reject a nondegenerate dynamic intertwiner and a
> deterministic empirical pushforward for the current objects. They do not
> exclude a newly source-produced random-scan kernel, a different common
> reference, or an independently justified stochastic coupling. Merely
> inventing such a coupling would not be source evidence.

The declared kernel is exactly such an invented coupling: it inhabits the
premise slot with one explicit law and is not source evidence.

## 3. Verified statements and numbers

All rationals in the receipt are exact strings; floats carry the tolerance
`1e-9` and are cross-checks only.

### Sectors, absorbing sets, minimal sets

- `s = 3`: 364 states, 220 absorbing (the agreement shell, loads in `{0, 1}`),
  144 transient. `V` ranges over `[3, 9]`; the `V`-minimal set has 220 states
  and equals the absorbing set. `Phi` ranges over `[9, 45]`; the `Phi`-minimal
  set has 20 states (the twenty faces, `Phi = 9`) and is a proper subset of
  the absorbing set, on which `Phi` takes the values 9 (20 states), 11 (60),
  13 (120) and 15 (20).
- `s = 4`: 1365 states, 495 absorbing, 870 transient; `V` in `[4, 16]` with
  495 `V`-minimal states equal to the absorbing set; `Phi` in `[10, 80]`, 30
  `Phi`-minimal states; `Phi` on the absorbing set: 10 (30), 12 (60), 14
  (180), 16 (195), 18 (30).
- Why the absorbing set is the `V`-minimal set: a state is absorbing exactly
  when every seam gap is at most one, that is, when every seam split is the
  floor/ceil split, the integer minimizer of `x_c^2 + x_o^2` at fixed
  endpoint total. On both sectors this seam-local condition coincides with
  global `V`-minimality (loads in `{q, q+1}` for `s = 12 q + r`), checked
  exhaustively. Agreement of the local and the global condition at larger
  totals is outside the verified scope: a state with load 0 on one port, 2 on
  its antipode and 1 elsewhere has every seam gap at most one and is not
  `V`-minimal at total twelve.

### Absorption

Expected number of moves to absorption from the uniform initial law over the
sector (waits count as moves), exact via the ascending-`V` recursion (the
transient support graph is acyclic), cross-checked by a float
fundamental-matrix solve (relative error below `1e-15`) and, at `s = 3`, a
second time by exact Gaussian elimination in the verifier. `beta = 0` is the
same uniform kernel for both potentials.

- `squared_norm_V`, `s = 3`: `beta = 0`: `261/91 = 2.868132` (maximum `27/2`
  from `(3, 0, ..., 0)`); `beta = 1`: `1467/1456 = 1.007555` (maximum
  `63/16`); `beta = 2`: `13311/23296 = 0.571386` (maximum `621/256`). Row
  normalizers reach 60, 210 and 2610.
- `squared_norm_V`, `s = 4`: `beta = 0`: `2627/455 = 5.773626` (maximum
  `37/2` from `(4, 0, ..., 0)`); `beta = 1`: `137299/70720 = 1.941445`
  (maximum `1445/256`); `beta = 2`: `14938487/13844480 = 1.079021` (maximum
  `239621/65536`). Row normalizers reach 60, 2610 and 655410.
- `seam_sum_Phi`, `s = 3`: `beta = 1`: `36040360761/83969966080 = 0.429205`;
  `beta = 2`: `0.428572` (exact in the receipt). `s = 4`: `beta = 1`:
  `0.802183`; `beta = 2`: `0.800002` (exact in the receipt). The seam-sum
  tilt is much steeper (normalizers up to `2^48` scale) because its
  decrements depend on the neighbour loads.
- Terminal states. Under `V` the terminal state is `V`-minimal with
  probability one from every state and the expected terminal `V` is `3`
  (`s = 3`) and `4` (`s = 4`) exactly. Under both potentials the probability
  that the terminal state is `Phi`-minimal from the uniform initial law is a
  diagnostic: `squared_norm_V`: `0.153846` at `s = 3` for every `beta` (at
  `s = 3` every active label of a state carries the same `V` decrement, so
  the tilt rescales only the waiting probability), `0.130220`, `0.130349`,
  `0.130389` at `s = 4`; `seam_sum_Phi`: `0.153846`, `0.213187`, `0.241112`
  at `s = 3` and `0.130220`, `0.259731`, `0.317334` at `s = 4`.

### Derived action and gauge (label paths on `s = 3`)

Initial states `A = (3, 0, ..., 0)`, `B = 2 e_0 + e_1` (port 1 adjacent to
port 0), `C = 2 e_0 + e_3` (port 3 the antipode of port 0), initial law `1/3`
each. For both potentials and every `beta`:

- All `3 x 216000` label paths of length three: the product of the drawn
  weights equals `2^(-beta sum dU)` on every path, so
  `S = beta ln 2 sum_k dU_k + sum_k ln Z(x_{k-1})`; the path law sums to one
  from each initial state; the tilt normalizer equals the Lean partition
  constant `60^(-3) = 1/216000`; the ratio of the path law to `exp(-S)` is the
  constant `1/3` on every path (the fitted action `S_fit = -log law` differs
  from `S` by the constant `ln 3`).
- Gauge characterization on all `3 x 3600` label paths of length two: `S + c`
  with `exp(-c) = 3` and `S/2` at multiplier two reproduce the law; `S + f`
  with `exp(-f) = 2` on paths whose first label is label 0 differs on all
  10800 paths; `2S` at multiplier one differs on all paths for `beta = 1, 2`
  (10 distinct action values under `V`, 13 under `Phi`) and reproduces the law
  for `beta = 0`, where the action is constant, in line with the nonconstancy
  hypothesis of `bare_log_action_multiplier_unique_of_nonconstant`. In every
  case the tilt equals the law exactly when the multiplier-weighted difference
  is constant.
- Sample exhibits per potential, `beta` and initial state store the labels,
  the states, the potential along the path, the decrements, the row
  normalizers, the `ln 2` coefficient, the exact path probability and the
  float value of `S`.

### Most probable paths (lengths one to four)

Argmax of the label path probability equals argmin of `S` at every length
from every initial state, compared as exact rationals. Ties:

- `beta = 0`: every label path of a given length ties (`60^n` paths).
- `squared_norm_V`, `beta = 1, 2` from `A`: length one, 10 tied label paths
  over 10 terminal states (`V = 5`); length two, 80 tied paths over 20
  terminal states, all `V`-minimal (`V = 3`, witness `Phi = 9`); length three,
  4800 paths; length four, 288000 paths (the additional steps are waits).
  From `B`: 8, 480, 28800, 1728000 tied paths over 4 terminal states. From
  `C`: 10, 600, 36000, 2160000 tied paths over 5 terminal states.
- `seam_sum_Phi`, `beta = 1, 2` from `A`: 10, 40, 2400, 144000 tied paths
  (5 terminal states from length two, `Phi = 9`); from `B`: 4, 240, 14400,
  864000 over 2 terminal states; from `C`: 10, 600, 36000, 2160000 over 5
  terminal states with `Phi = 13`.

### Non-identifiability family

Transported definition: the per-step record is the occupancy vector `y` of
the sixty directed labels (one-hot on realized records, so each coordinate is
0 or 1 on realized histories), `L_0(x, y) = sum_m y_m (-log P(m | x))` is the
log-transition table extended affinely in the record slot, and
`L_a = L_0 + (a/2) sum_m y_m (y_m - 1)`. Verified for both potentials and
`a` in `{1, 2, 5/2}`:

- the correction is exactly zero on all sixty realized records and on all
  `3 x 3600` enumerated label paths of length two, so every member induces
  the same path law (the tilt by `L_a` equals the tilt by `L_0`);
- on the unrealized configuration `y_m = 2` the corrections are `1`, `2`,
  `5/2`; at `y_m = 1/2` they are `-1/8`, `-1/4`, `-5/16`, and the `a = 1`
  versus `a = 2` gap is `1/8`, the corpus's number.

Verified statement: selection inside the declared family from realized
histories is impossible for this kernel.

### Reversibility and entropy

- Every state-changing transition lowers `V`, so the support graph on
  transient states is acyclic (no cycle of length two or more) and no
  transition has a reverse: 720 support edges between distinct states at
  `s = 3` (4620 at `s = 4`), none with reverse support, for both potentials
  and every `beta`. A positive reference with `pi_x P_xy = pi_y P_yx` would
  force `P_xy = 0` on every such edge, so no reversible reference exists;
  stationary laws vanish on transient states, so no faithful stationary
  reference exists. The declared kernel is a strict-descent normalizer, which
  is the flagship's stated gap.
- Path entropy production `sum P(path) log(P(path)/P(reversed path))` is
  infinite (every state-changing path has a zero-probability reverse); the
  finite part over reversible pairs is zero. Diagnostic numbers: the one-step
  irreversible mass from the uniform law (`11/182` at `s = 3`, `134/1365` at
  `s = 4`, `beta = 0`) and the Shannon entropy trace of the state law from
  the uniform initial law, which falls from `8.508` bits to `7.726` bits under
  `V` for every `beta` at `s = 3` (and to `7.726`, `7.649`, `7.596` under
  `Phi` for `beta = 0, 1, 2`), and from `10.415` to `8.851`, `8.850`, `8.850`
  under `V` at `s = 4` (`8.851`, `8.563`, `8.394` under `Phi`); the relative
  entropy to the uniform reference rises. The strict-descent normalizer
  carries no entropy inequality.

## 4. Receipt and custody

`data/exact/phi_linked_kernel_receipt.json` is canonical JSON (sorted keys,
compact separators, ASCII, no NaN, trailing newline) with rationals as strings.
It pins by sha256 the producer, the verifier, the test file,
`oph_exact/carrier.py`, the Lean module `LogTransitionAction.lean`, the
observers paper tex and the flagship tex, all committed files. The
instantiation paper is an uncommitted working file of another repository and
is referenced without a hash in the `paper_reference` block (repository
`oph-meta`, path, title, the subsection name "(M5) A source-selected action",
`hashing: none`); edits to that paper leave the receipt valid.

    .venv/bin/python -m oph_exact.phi_kernel --write
    .venv/bin/python -m oph_exact.phi_kernel --check
    .venv/bin/python -m oph_exact.verify_phi_kernel_independent
    .venv/bin/python -m pytest tests/test_exact_phi_kernel.py -q

The verifier does not import the producer. It derives the seams from the
oriented face list, enumerates the sectors recursively, rebuilds the move law
and the kernels of both potentials, recomputes every recorded number (exact
values exactly, floats within `1e-9`), solves the `s = 3` expected moves a
second time by exact Gaussian elimination, and recomputes the pins.

## 5. Claim boundary

Declared: the kernel family with tilt base two, the potential axis
(`squared_norm_V` primary, `seam_sum_Phi` diagnostic), the betas, the
protected totals, the wait reading on gap-one seams, the initial law of the
path checks.

Verified: descent for both potentials, the flagship unit-transfer identity
and the declared-move closed form for `V`, absorbing set equal to the
`V`-minimal set, expected moves, derived action with the Lean partition
constant, gauge characterization, most-probable paths with tie counts, the
non-identifiability family, absence of a reversible reference.

Not claimed: source selection of the kernel, laboratory readout, a continuum
limit, an entropy inequality for the strict-descent normalizer, uniqueness of
the terminal state under the seam-sum variant, agreement of seam-local and
global `V`-minimality beyond the two verified totals. The kernel is declared,
so (M5) stays open; the receipt inhabits the flagship's missing
stochastic-coupling premise with one explicit law. A source derivation of the
tilt is work in progress in the theory repository.
