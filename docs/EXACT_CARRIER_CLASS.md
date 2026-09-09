# Exact carrier-class selection (lane L3)

Machine check of the carrier-selection proposition of the instantiation paper
(`trt-scspl/instantiating_the_self_configuring_self_processing_language_that_is_our_universe.tex`,
`prop:euler`, `prop:carrier`, and the dimension argument of `thm:forcing`).
The paper's remark states that the proposition is not machine-checked; this
lane supplies the check in exact arithmetic.

- Producer: `oph_exact/carrier_class.py` (`--write`, `--check`)
- Independent verifier: `oph_exact/verify_carrier_class_independent.py`
- Receipt: `data/exact/carrier_class_selection_receipt.json`
  (schema `oph.exact.carrier-class-selection.v1`)
- Tests: `tests/test_exact_carrier_class.py`

```
.venv/bin/python -m oph_exact.carrier_class --check
.venv/bin/python -m oph_exact.verify_carrier_class_independent
.venv/bin/python -m pytest tests/test_exact_carrier_class.py -q
```

## 1. The check

The paper narrows the carrier in three steps and this lane checks each of
them on the three uniform simplicial triangulations of the sphere.

1. **Euler count.** A simplicial triangulation of `S^2` with every vertex of
   degree `d` has `F = 4d/(6-d)`, `E = 3F/2`, `V = 3F/d`. Positivity forces
   `d < 6`; `d = 1` gives `F = 4/5`; `d = 2` gives the dihedron (two faces on
   the same three vertices, non-simplicial); `d = 3, 4, 5` give the
   tetrahedron `(4, 6, 4)`, the octahedron `(6, 12, 8)` and the icosahedron
   `(12, 30, 20)`. Uniqueness of each triangulation is classical and declared.
2. **Equivariance character constraint.** With `Rot(K)` the orientation-
   preserving automorphism group acting on the port set `P`, a complete
   reversible response `D: R^P -> u(H)` with the endogenous transport clause
   `Ad_{[U_a]} D(v) = D(a.v)` and inner implementers makes `D` an isomorphism
   of `Rot(K)`-modules onto `g = D(R^P)`. Inner automorphisms preserve every
   simple ideal and fix the centre pointwise, so `g = z + s_1 + ... + s_k`
   with `dim g = |P|`, `dim z <= dim (R^P)^{Rot(K)} = 1`, each `s_i` carrying
   a homomorphism of `Rot(K)` into its adjoint group, and the character
   identity `chi_P = dim z + sum_i chi_{Ad o rho_i}` on every class.
3. **Equivariant inverse pairing.** A fixed-point-free involution of `P`
   commuting with `Rot(K)`.

## 2. Method

Every decision is taken over the integers, `Fraction`, or `Q(sqrt 5)` pairs
`(a, b)` meaning `a + b sqrt 5`. No floating point enters.

- **Complexes.** Oriented face lists; the icosahedron is the committed
  carrier of `oph_exact/carrier.py`, the tetrahedron and the octahedron are
  built in the producer. Each complex is checked to be simplicial, closed and
  consistently oriented (every directed edge once in each direction), with
  cyclic vertex links, connected, of uniform degree and Euler characteristic
  two, and to satisfy the count formulas.
- **Groups.** Incidence automorphisms by backtracking on adjacency, the
  rotation group as the subgroup preserving the cyclic order of every
  oriented face (index two). Closure, identity, inverses and transitivity are
  checked. Conjugacy classes carry element order, fixed-vertex count, an axis
  witness (swapped edge, invariant face, fixed vertex pair at maximal
  distance) and, for order five, the neighbour step that separates the
  `2 pi / 5` and `4 pi / 5` rotations. Class keys are invariant-based
  (`o2f0` = order 2, no fixed vertex; `o5f2s1` = order 5, two fixed vertices,
  neighbour step 1), so the producer and the verifier compare data across
  different vertex labellings.
- **Characters.** The constituents of `R^P` are derived, without any
  declared table, as the characters of the adjacency eigenspaces: minimal
  polynomial of the adjacency matrix over `Q`, roots in `Q(sqrt 5)`, exact
  Lagrange projectors, `chi_k(g) = tr(P_k M_g)`. Each band has norm one, so it
  is irreducible. The real character table is closed under tensor, symmetric
  and exterior squares (a remainder of norm one with Frobenius-Schur indicator
  one is a real-type irreducible, norm two with indicator zero a complex-type
  one), until `sum dim^2 / norm = |G|`, and orthogonality is checked on the
  closed table. The verifier takes the opposite route: declared standard
  tables of `A4`, `S4`, `A5`, verified against its own recomputed groups by
  orthogonality, completeness, Frobenius-Schur indicators and integrality of
  the ring operations.
- **Lie candidates.** Partitions of `|P| - dim z`, `dim z in {0, 1}`, into
  parts from the declared table `su(2) = 3`, `su(3) = 8`, `so(5) = 10`
  (the next simple algebra, `g2`, has dimension 14).
- **Adjoint homomorphisms.** `Rot(K) -> SO(3)`: three-dimensional real
  orthogonal representations with trivial determinant (top exterior power
  computed from the class power maps), adjoint character the exterior
  square, which equals the representation character. `A5 -> PSU(3)`: the
  three-dimensional representations `1+1+1`, `3`, `3'`, adjoint character
  `|chi|^2 - 1`; the lift to `SU(3)` is declared (`H^2(A5, Z/3) = 0`).
- **Assignment test.** For every candidate and every multiset of
  homomorphisms per ideal type, the identity `chi_P = dim z + sum chi_Ad` is
  tested on every class; failures record the first failing class and, where
  one class refutes every assignment at once, the uniform witness.
- **Pairings.** Exhaustive enumeration of the perfect matchings of `P`
  (`3`, `15`, `10,395`) tested for commutation with every rotation, and the
  centralizer of the rotation group in `Sym(P)` (determined by the image of
  one vertex under transitivity); the two enumerations are required to agree.

The verifier builds the groups by flag propagation (a rotation is fixed by
the image of one oriented face and spreads face by face), rebuilds the
receipt's own face lists as well, and verifies the labelled objects of the
receipt (class representatives, pairings, centralizers, reflection) against
them.

## 3. Results per carrier

### Tetrahedron, `Rot = A4`, order 12 (incidence automorphisms 24)

| class | name | size | order | fixed | `chi_P` |
|--|--|--|--|--|--|
| `o1f4` | identity | 1 | 1 | 4 | 4 |
| `o2f0` | half turn about an edge-midpoint axis | 3 | 2 | 0 | 0 |
| `o3f1a` | third turn about a vertex-face axis | 4 | 3 | 1 | 1 |
| `o3f1b` | third turn about a vertex-face axis (inverses) | 4 | 3 | 1 | 1 |

Real irreducible characters: `1 = (1, 1, 1, 1)`, `2c = (2, 2, -1, -1)`
(complex type, the pair of nontrivial linear characters), `3 = (3, -1, 0, 0)`
(adjacency eigenvalue `-1`). `R^P = 1 + 3`.

Candidates: `u(1)+su(2)` only (a vanishing centre would need a semisimple
algebra of dimension four, of which there is none). Homomorphisms
`A4 -> SO(3)`: `3` (fixed dimension 0), `1+2c` (fixed dimension 1), `1+1+1`
(fixed dimension 3). Assignment outcomes: `3` survives, `1 + (3, -1, 0, 0) =
(4, 0, 1, 1) = chi_P`; `1+2c` and `1+1+1` fail on `o2f0`. Pairing: three
perfect matchings (the three double transpositions), none commutes with `A4`;
the centralizer of `A4` in `Sym(4)` is trivial. Verdict: consistent with
`u(1)+su(2)` on the vector representation, and excluded by the pairing.

### Octahedron, `Rot = S4`, order 24 (incidence automorphisms 48)

| class | name | size | order | fixed | `chi_P` |
|--|--|--|--|--|--|
| `o1f6` | identity | 1 | 1 | 6 | 6 |
| `o2f2` | half turn about a vertex axis | 3 | 2 | 2 | 2 |
| `o2f0` | half turn about an edge-midpoint axis | 6 | 2 | 0 | 0 |
| `o3f0` | third turn about a face axis | 8 | 3 | 0 | 0 |
| `o4f2` | quarter turn about a vertex axis | 6 | 4 | 2 | 2 |

Real irreducible characters: `1`, `sgn = (1, 1, -1, 1, -1)`,
`2 = (2, 2, 0, -1, 0)` (adjacency eigenvalue `-2`), `3 = (3, -1, -1, 0, 1)`
(adjacency eigenvalue `0`, the vector representation), `3' = (3, -1, 1, 0, -1)`.
`R^P = 1 + 2 + 3`.

Candidates: `su(2)+su(2)` only (a one-dimensional centre would leave
dimension five, of which there is none). Homomorphisms `S4 -> SO(3)`: `3`,
`sgn+2`, `1+1+1`, `1+sgn+sgn`; rejected with determinant `sgn`: `3'`, `1+2`,
`1+1+sgn`, `sgn+sgn+sgn`. Assignment outcomes: ten multisets, none survives.
Uniform witness: on the edge half-turn class `o2f0` the permutation character
is `0` while every candidate trace lies in `{-2, 2, 6}` (each homomorphism
contributes `3` or `-1` there). The trivial-multiplicity count alone does not
exclude the octahedron (the values `{0, 1, 2, 3, 4, 6}` contain `1`); the
trace witness does. Pairing: fifteen perfect matchings, one commutes (the
antipodal map, graph distance two), centralizer of order two. Verdict:
excluded by A1-A2 equivariance.

### Icosahedron, `Rot = A5`, order 60 (incidence automorphisms 120)

| class | name | size | order | fixed | `chi_P` |
|--|--|--|--|--|--|
| `o1f12` | identity | 1 | 1 | 12 | 12 |
| `o2f0` | half turn about an edge-midpoint axis | 15 | 2 | 0 | 0 |
| `o3f0` | third turn about a face axis | 20 | 3 | 0 | 0 |
| `o5f2s1` | fifth turn about a vertex axis, `2 pi / 5` | 12 | 5 | 2 | 2 |
| `o5f2s2` | fifth turn about a vertex axis, `4 pi / 5` | 12 | 5 | 2 | 2 |

Adjacency spectrum: `5` (dimension 1), `sqrt 5` (dimension 3, the vector
representation `3`, which is the flagship's slow Laplacian band `5 - sqrt 5`),
`-1` (dimension 5), `-sqrt 5` (dimension 3, `3'`). Real irreducible
characters with `phi = (1 + sqrt 5)/2`:

| irrep | `o1f12` | `o2f0` | `o3f0` | `o5f2s1` | `o5f2s2` |
|--|--|--|--|--|--|
| `1` | 1 | 1 | 1 | 1 | 1 |
| `3` | 3 | -1 | 0 | `phi` | `1 - phi` |
| `3'` | 3 | -1 | 0 | `1 - phi` | `phi` |
| `4` | 4 | 0 | 1 | -1 | -1 |
| `5` | 5 | 1 | -1 | 0 | 0 |

`R^P = 1 + 3 + 3' + 5`, with `1 + phi + (1 - phi) + 0 = 2` on both order-five
classes and `1 - 1 - 1 + 1 = 0` on the edge half-turns.

Candidates: `su(2)^4` (centre zero) and `u(1)+su(2)+su(3)` (centre one; the
unique split of eleven is `3 + 8`). Homomorphisms `A5 -> SO(3)`: `3`, `3'`,
`1+1+1`. Homomorphisms `A5 -> PSU(3)`: `3` with adjoint character `3 + 5`,
`3'` with `3' + 5`, `1+1+1` with `8 . 1`.

Assignment outcomes:

- `su(2)^4`: fifteen multisets, none survives; the trivial multiplicities take
  the values `{0, 3, 6, 9, 12}` against the required `1` (the fixed
  dimension of each factor is zero or three, the forcing theorem's argument).
- `u(1)+su(2)+su(3)`: nine assignments, two survive, `su(2)` on `3` with
  `su(3)` on `3'+5`, and `su(2)` on `3'` with `su(3)` on `3+5`. The outer
  automorphism of `A5` (conjugation by a transposition of the five Klein
  four-subgroups) swaps `o5f2s1` with `o5f2s2` and `3` with `3'`, and
  exchanges the two survivors. Conjugation by an incidence reflection is
  inner (the full group is `A5 x Z2`), so no isometry realizes the swap.

Pairing: `10,395` perfect matchings, exactly one commutes with the sixty
rotations, the antipodal map at graph distance three, equal to
`carrier.antipode()`; the centralizer of `A5` in `Sym(12)` has order two.
Verdict: `u(1)+su(2)+su(3)`, with an equivariant inverse pairing.

## 4. Summary and declared inputs

Within the uniform spherical triangulations, A1-A2 equivariance excludes the
octahedron, the equivariant inverse pairing excludes the tetrahedron, and the
icosahedron carries `u(1)+su(2)+su(3)` with `su(2)` on the vector
representation up to the outer automorphism of `A5`.

Declared (supplied, unchecked here):

- the carrier class: simplicial triangulations of the sphere of uniform
  vertex degree, and the classical uniqueness of the `d`-regular one for
  `d = 3, 4, 5`;
- the classification table of compact simple Lie algebras of dimension at
  most twelve;
- inner implementers of the A2 transport clause, hence the factorization of
  the action through the adjoint group of each ideal and the pointwise fixed
  centre;
- the identification of homomorphism classes with representation classes
  (`SO(3)`: orthogonal representations of trivial determinant; `PSU(3)`: the
  lift of every homomorphism of `A5` to a linear representation);
- the hypotheses of the proposition: existence of the complete reversible
  response `D` of A1 and of the transport clause of A2.

Not claimed: a derivation of the carrier class from records; a Lean check of
the proposition (work in progress); the proof of the flagship forcing
theorem, whose character-constraint outcome the receipt re-derives on the
icosahedron; any selection outside the uniform spherical triangulations.

## 5. Receipt conventions

Canonical JSON (`sort_keys`, compact separators, ASCII, no non-finite
constants, trailing newline). Fractions are strings, `Q(sqrt 5)` values are
`["a", "b"]` meaning `a + b sqrt 5`. `implementation_pins` carries sha256 pins
of the producer, the verifier, the test file and `oph_exact/carrier.py`. The
paper source lives in the oph-meta repository and carries no hash;
`paper_reference` records its repository-relative path, title, the two
proposition names and the forcing theorem label.
`receipt_sha256` is the digest of the receipt without that key. The `scope`
block lists the declared inputs and the items not claimed; `claim_boundary`
states the result in one sentence.
