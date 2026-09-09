# Exact closure loop at carrier scale (lane L4)

Producer `oph_exact/closure_loop.py`, verifier
`oph_exact/verify_closure_loop_independent.py`, tests
`tests/test_exact_closure_loop.py`, receipt
`data/exact/closure_loop_receipt.json`, event logs
`data/exact/closure_loop_logs/`.

## The condition being run

The flagship's closure hypothesis requires the specification observers recover
from their records, the realization they construct from that specification,
and the structure they inhabit to return the same invariant quantities. The
instantiation paper reads this as M=R: the recovered specification is the
theory, the constructed realization is the universe, and the closure condition
identifies them. This lane runs that condition on one exact carrier, where
every object is finite and every quantity is computable exactly.

## Three stages and the firewall

Stage A, the generator, runs a carrier under a repair law with the uniform A3
seam schedule from seeded integer loads. It emits only an event log:

```
{"schema": ..., "ports": n, "carriers": 1, "ports_per_carrier": n,
 "initial": [...], "events": [{"step": k, "changed": {port: [before, after], port: [before, after]}}, ...],
 "final": [...], "probe": {"steps": [1, 5, 30, 100, 300], "readback_digits": 120, "readback": {...}}}
```

Readings are exact rationals written as strings. The probe is the readback
after `2n` synchronous expectation steps from each one-hot state, rounded to
120 significant decimal digits (the declared instrument resolution). No seam
list, group, orientation, law description or schedule description enters the
log. The federation log carries a carrier index per event and global port
ids.

Stage B, the recovery, is one fixed algorithm with declared constants. It
reads the log and nothing else, the same algorithm on every log:

1. the port count;
2. the seam set as the port pairs an event touches, with the check that every
   event touches exactly two ports; seam count and degree sequence;
3. the inverse-port pairing as graph distance three when that map is a
   fixed-point-free involution, else `none`; the farthest-port pairing is
   reported beside it;
4. the incidence automorphism group by backtracking over the recovered graph,
   the triangles as 3-cycles, one consistent orientation up to global
   reversal (A1 declares which of the two classes holds), and the
   orientation-preserving subgroup for both classes;
5. the repair rule, classified from conservation, symmetry and form of every
   event: `seam_mean`, `integer_nearest_agreement` (with the empirical tie
   placement frequency), `overwrite` (one side unchanged, the other copied) or
   `unclassified`;
6. the schedule law by the exact chi-square statistic of the seam event counts
   against `1/|seams|` at significance `1e-3`;
7. the response Gram from the probe: centre and normalize the readback into
   `K_n = m C_n / tr C_n`, eigenvalues and rank with tolerance `1e-9` at the
   largest step whose centered kernel keeps at least twelve significant digits
   of the readback; the probe is also predicted from the recovered seam set
   with `T_rec = I - L_rec / (2 |seams|)` and compared at the readback
   tolerance;
8. the Lie type dimension count: the permutation character of the recovered
   rotation group, its trivial-isotypic dimension, the exact Laplacian bands
   over `Q(sqrt 5)` with their characters, and the surviving splits of the port
   count into a centre of dimension at most one plus simple dimensions from
   the declared table `{3, 8, 10}`: a 3 must carry the group through `SO(3)`
   (determinant one on the band, computed from the power sums of the exact
   character), an 8 must carry the adjoint character `chi_rho^2 - 1` of some
   such 3-dimensional `rho`, a 10 the adjoint character of a 5-dimensional
   special orthogonal band;
9. the descent law and the terminal law: strict descent of the declared
   potential over the log's events, and the terminal state of the recovered
   rule on the recovered seams from the log's initial loads over eight
   replayed schedules.

Stage C, the instantiation, builds a carrier from the recovered seams, rule
and schedule alone and Stage A runs it with a fresh seed. Stage B on the source
log gives the inhabited-structure invariants; Stage B on the instantiated run
gives the constructed-realization invariants. The loop iterates once more from
the realization's own log.

## The invariant vector

Ports, seam count, degree sequence, pairing type, farthest pairing distance,
automorphism and rotation orders, rule class, presence of nonconservative
events, tie law, schedule law, Gram eigenvalues (rounded to `1e-9`), Gram
rank, Gram step index, probe prediction verdict, Lie split with its assignment
count, trivial-isotypic dimension, strict-descent violation count, and the
terminal law (terminal equals the initial mean, schedule independence,
quotient-multiset schedule independence, terminal invariant type), plus the
component count, component isomorphism and the global schedule law.

Equality of the vectors across the three stages, together with a terminal
invariant of type `state` or `quotient_multiset`, is the receipt
`CLOSURE_FIXED_POINT_AT_CARRIER_SCALE`.

## Outcomes

Canonical source (icosahedron, seam mean): 12 ports, 30 seams, degree five,
the distance-three involution equal to the carrier's antipode, groups 120 and
60, rule `seam_mean` with 360 conservative symmetric events, chi-square
`181/6` against the threshold `58.3` at 29 degrees of freedom, Gram
eigenvalues `4, 4, 4, 0 x 9` at `n = 300` (rank 11 at every smaller step),
kernel equal to `4 P_slow`, Lie split `1+3+8` with two assignments related by
the outer automorphism (the su(2) factor on either 3-band, the su(3) factor on
the complementary `5 + 3'`), band characters `phi` and `1 - phi` on the two
order-5 classes, zero strict-descent violations, terminal state equal to the
initial mean on all eight schedules. The two instantiated runs return the same
vector: the fixed point holds at the state level.

Integer source: rule `integer_nearest_agreement`, tie frequency `48/89` on
the source log and `51/98`, `39/70` on the realizations, tie law uniform, the
terminal state schedule dependent and its sorted multiset schedule
independent: the fixed point holds on the quotient invariants.

Overwrite source: rule `overwrite`, nonconservative events present, positive
strict-descent violation counts that differ between runs, terminal state
schedule dependent, no terminal invariant, no fixed point.

Tetrahedron: 4 ports, 6 seams, groups 24 and 12, no pairing at any distance,
Gram rank 3 with eigenvalues `4/3`, Lie split `1+3`. Octahedron: 6 ports, 12
seams, groups 48 and 24, no distance-three pairing, the antipodal pairing at
distance two, Gram rank 3 with eigenvalues `2`, no surviving Lie split (the
only candidate `3+3` fails because the edge half-turns act with determinant
minus one on the `1 + 2` block). Gram rank alone does not select the
icosahedron; the pairing together with the rotation order does, and the Lie
split separates all three.

Isolated federation of twenty icosahedral carriers: the recovery returns
twenty isomorphic components with the single-carrier invariant vector, the
global schedule uniform over the six hundred seams, and a second instantiation
round agreeing. The probe steps of a federation log scale with the carrier
count because the synchronous expectation of one event among all seams damps
each carrier twenty times slower.

## Two findings recorded by the lane

The seam Laplacian form `sum_seams (x_i - x_j)^2` is not an event-wise
Lyapunov function of the seam-mean law: moving one endpoint toward its partner
can move it away from its other neighbours, and the count of such increases is
reported for every log. The declared descent potential is the centered squared
norm `|x - mean(x) 1|^2`, which every conservative seam retraction decreases
strictly whenever the multiset of the two readings changes. This is the
flagship's quadratic descent functional: the integer record keeping paragraph
of `from_observer_consensus_to_standard_physics.tex` (lines 889 to 895 at RER
r2039) declares `V(N) = sum_i N_i^2` and proves that a conservative unit
transfer across a seam with oriented mismatch `d >= 2` decreases `V` by
exactly `2(d - 1)`; the confluence theorem's "exact quadratic descent" is this
`V`. On a conservative law the centered squared norm equals `V` minus the
conserved `(sum_i x_i)^2 / ports`, so the two potentials have the same
decrements (a seam-mean event with mismatch `d` decreases both by `d^2 / 2`).
The receipt carries the citation as `parameters.potential_reference` and in
every recovered descent block.

A finite readback resolution bounds the probe. At 48 digits the tetrahedron's
centered kernel at `n = 300` (relative size `(2/3)^600`) rounds to zero; the
declared 120-digit readback keeps fourteen digits of it and the recovery reads
the Gram at the largest resolved step.

## What a universe-level closure experiment needs in addition

The glued federation with the inter-carrier gluing of lane L1
(`oph_exact/federation.py`), so that the recovered specification includes the
gluing convention and the receipt records whether the rank-three band survives
it; this is work in progress. Records as the only observer access: the probe
of this lane is a direct readback of the carrier state, and a universe-level
loop has to recover the same invariants from committed records alone.
Refinement across levels, so that the recovered specification at one level
constrains the specification at the next. Existence and uniqueness of the
cosmic fixed point, and the physical identification of any invariant, are
outside every receipt of this lane.
