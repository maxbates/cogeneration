Spec: Variable-length motif scaffolding via tree-encoded birth events + Brownian / IGSO(3) bridges

NOTE: spec.md contains legacy detailed notes; plan.md is the source of truth. This repo currently follows the “fork” variant: split head predicts remaining DIRECT children; deletion is inference-only with fixed hazard on to-be-deleted nodes.

Goal

Train a protein generator that co-generates sequence + structure (AF2-style frames) for variable-length scaffolds around fixed motifs, tolerating large insertions (≈3–100 residues), using:
	•	a binary “insertion tree” over scaffold residues (implicit ancestry),
	•	per-residue birth times (when a residue becomes “alive”),
	•	optional to-be-deleted dummy residues (length augmentation),
	•	Brownian bridge for translations and IGSO(3) for rotations.

This is a simplified instantiation of Branching Flows ideas: model outputs a base prediction plus per-element split signals; splits use learned "remaining children count" (x1-prediction) with hazard h(t)=1/(1-t), and deletions use a fixed hazard h_del(t)=1/(1-t) for designated to-be-deleted nodes.

⸻

Relationship to Branching Flows

Branching Flows defines a variable-length process where elements split (duplicate) and delete with rates driven by (learned) “remaining splits” and “deletion probability,” and uses a conditional path construction so you can sample intermediate x_t given the endpoint.  ￼

This spec implements a practical version for proteins:
	•	We encode the split history as arrays (parent_idx, birth_time) rather than an explicit tree object.
	•	We can optionally augment data with to-be-deleted residues by duplicating residues and inserting the duplicates next to the originals. These are eligible for deletion via event-driven hazard sampling during trajectory.
	•	During training corruption, residues with t < birth_time are treated as not yet present (masked out), which matches the semantics of "variable length over time."

⸻

Definitions

Alignment length

For each training example:
	•	Start with a protein of length N.
	•	Sample motifs (mask) of total length M (these are “always present,” fixed/conditioned).
	•	Remaining scaffold residues are S = N - M.
	•	Optionally add D “dummy-to-delete” residues, producing an alignment length
A = N + D.
All tensors in the new interpolant use length A (with masks controlling what is “alive”).

State per residue

Initially keep the same modalities as your existing fixed-length system:
	•	translations trans in \mathbb R^3,
	•	rotations rotmats in SO(3),
	•	torsions (optional; can be deferred),
	•	sequence aatypes (categorical).

You can start with CA-only (translations only) and add rotations later; the spec supports both.

⸻

New per-example latent coupling Z (stored as batch features)

We represent the binary forest + event times using arrays of length A:

Required arrays (per example)
	•	parent_idx: int64[A]
	•	For each non-motif residue, parent_idx[i] is the index of its parent that existed earlier.
	•	For motif roots, parent_idx[i] = i (or -1), and birth_time[i]=0.
	•	birth_time: float32[A] in [0,1]
	•	Motifs: 0.
	•	Scaffold residues and dummy residues: sampled subject to birth_time[parent] < birth_time[i].
	•	is_motif: bool[A]
	•	to_be_deleted: bool[A] - designates which residues are eligible for event-driven deletion
	•	Note: deleted_mask is a dynamic state during sampling only (not stored in batch, not used in training)
	•	Note: deleted_mask tracks which nodes have been deleted by events during sampling

Optional helper arrays (recommended)
	•	topo_order: int64[A] indices sorted by increasing birth_time (ties broken with parent-first)
	•	gap_id: int32[A] if you have multiple motif gaps (to enforce locality of parents within a gap)
	•	chain_id, res_idx_align: if needed for multimers / positional encoding

This array encoding is the tree; a helper can materialize children lists for debugging, but training/sampling should not need explicit trees.

⸻

How to sample the binary insertion tree (non-midpoint-biased)

For each gap interval between motif boundary residues in the alignment order:
	1.	Let the final scaffold indices in that gap be I = [i0, i1, ..., i(L-1)] in final left-to-right order.
	2.	Build a random local binary tree over the interval:
	•	Maintain a set of active sub-intervals.
	•	Pick a sub-interval, pick a split point biased away from midpoint (e.g., choose split index with distribution proportional to distance-to-endpoints or a Beta-biased fraction), and recurse.
	3.	Assign each created node a parent as one of its adjacent already-created neighbors (enforcing neighbor-only splits in the final order).

This produces parent_idx with strong locality without forcing a balanced tree.

Birth time schedule

Sample birth times with a controllable “early/late” bias:
	•	Sample u ~ Beta(α, β) per non-motif node
	•	Set birth_time = clamp(u, ε, 1-ε)
	•	Enforce causality: birth_time[i] = max(birth_time[i], birth_time[parent]+ε) in topo order

⸻

Training corruption: single-pass creation-state DP + Brownian/IGSO(3) bridges

High-level

Given a training example with endpoint X1 (the data protein with motifs embedded and optionally dummy residues inserted) and latent coupling arrays, we sample t ~ Uniform(0,1) and construct a corrupted X_t:
	•	A residue is alive if t >= birth_time[i]
	•	IMPORTANT: Deletion events are NOT simulated during training corruption
	•	deleted_mask is used only during sampling (generation), not during training
	•	For alive residues, sample the state at time t using a bridge from its creation state at birth_time[i] to its final state at t=1.

The only nontrivial part is computing all creation states C[i] = X_i(birth_time[i]), because for non-root nodes:
C[i] = X_{parent(i)}(birth\_time[i]).
This is computed in one topological pass (no recursion at query time).

Single-pass algorithm (per example)

Inputs: X1 (trans_1, rotmats_1, aatypes_1, masks), arrays parent_idx, birth_time, topo_order, sampled scalar t.
	1.	Initialize C_trans[i], C_rot[i], C_aa[i] for all i.
	2.	For motif roots: set C_*[i] = motif_init(i) (usually just equal to X1 motif state, or a fixed prior at t=0 for unconditional motifs).
	3.	Traverse i in topo_order:
	•	If is_motif[i]: continue
	•	Let p = parent_idx[i], s = birth_time[i]
	•	Compute parent state at time s by bridging from parent’s creation state to parent’s endpoint:
	•	x_p(s) ~ BrownianBridge(C_trans[p], trans_1[p], s_p=birth_time[p], t=s)
	•	R_p(s) ~ IGSO3Bridge(C_rot[p], rotmats_1[p], s_p=birth_time[p], t=s)
	•	Set child creation state as a split-copy:
	•	C_trans[i] = x_p(s) and C_rot[i] = R_p(s)
	•	C_aa[i] = <mask token> (or optionally parent AA at s, but masking is simpler)
	4.	Construct alive mask:
	•	alive[i] = (t >= birth_time[i])
	5.	For each i:
	•	If not alive: set trans_t[i]=0, rotmats_t[i]=I, aatypes_t[i]=MASK, and diffuse_mask_t[i]=0 (or keep a separate alive_mask)
	•	If alive:
	•	trans_t[i] ~ BrownianBridge(C_trans[i], trans_1[i], s=birth_time[i], t=t)
	•	rotmats_t[i] ~ IGSO3Bridge(C_rot[i], rotmats_1[i], s=birth_time[i], t=t)
	•	aatypes_t[i] from your discrete corruption schedule (mask/unmask / DFM-like), conditioned on “born at s”

Outputs: a NoisyFeatures dict with the same keys your model already expects (trans_t, rotmats_t, aatypes_t, plus time scalars), and additional masks/arrays needed for the model to understand variable length.

Bridge kernels
	•	Translations: Brownian bridge closed-form (mean is linear between endpoints; variance scales as (t-s)(1-t)/(1-s)).
	•	Rotations: IGSO(3) bridge sampler. (Implementation may use “geodesic mean + IGSO3 noise with effective time.”)

⸻

Split supervision target (x1-prediction)

**"Remaining direct children" count at time t**

For a binary tree, compute the per-node count:
	•	remaining_direct_children[i] = count of DIRECT children j where birth_time[j] > t
	•	This is ∈ {0, 1, 2} for a binary tree (not descendants—just immediate children)
	•	Computed from child1_idx, child2_idx arrays by checking which children are unborn

This is the x1-prediction target from Branching Flows: at time t, predict how many direct children this node will spawn by t=1.

**Deletion (event-driven, no learned head)**

Deletions are sampled as events during trajectory (sampling only), not predicted by a learned head:
	•	to_be_deleted[i] flag (from augmentation) designates which residues are eligible for deletion
	•	No learned deletion head or deletion loss during training
	•	During sampling only: deletion hazard λ_del = h_del(t) = 1/(1-t) for to_be_deleted nodes
	•	Deletion events sampled as Bernoulli(1 - exp(-λ_del * Δt)) at step boundaries
	•	When deletion fires: set deleted_mask[i] = True (drops from attention immediately)
	•	With exploding hazard h_del(t)=1/(1-t), to_be_deleted nodes are almost surely removed by t=1

⸻

Batch / feature interface

Reuse existing keys

The new interpolant should output the same keys your model already consumes:
	•	NoisyBatchProp.r3_t, .so3_t, .cat_t (scalar times)
	•	NoisyBatchProp.trans_t, .rotmats_t, .aatypes_t, optionally .torsions_t
	•	BatchProp.res_mask, .diffuse_mask, .motif_mask (now length A)

Add new keys (new enum or raw string keys)

Add a small "TreeBatchProp" namespace, e.g.:
	•	tree_parent_idx : (B, A) int64
	•	tree_birth_time : (B, A) float32
	•	tree_to_be_deleted : (B, A) bool (which residues are eligible for deletion)
	•	tree_child1_idx : (B, A) int64 (first child, -1 if none)
	•	tree_child2_idx : (B, A) int64 (second child, -1 if none)
	•	tree_is_motif : (B, A) bool (or reuse motif_mask)
	•	tree_alive_mask : (B, A) bool at sampled t
	•	Note: deleted_mask is sampling-only dynamic state (not stored in training batches)

Model should use tree_alive_mask (or updated res_mask) to avoid attending to nonexistent residues.

During training: alive_mask = (t >= birth_time)
During sampling: alive_mask = (birth_time < inf) & ~deleted_mask

⸻

Sampling process (forward generation)

Forward sampler with online births and event-driven deletion:
	1.	Sample motifs and their fixed states (condition input).
	2.	Sample a tree topology (parent_idx, child indices) and to_be_deleted flags from your prior (gap-wise).
	3.	Initialize state at t=0:
	•	motif residues present with fixed states,
	•	non-motif residues: birth_time = +inf (unborn), deleted_mask = False (dynamic state, sampling-only)
	4.	For step times t_k -> t_{k+1}:
	•	Run model forward on alive residues (attn_mask = alive_mask = (birth_time < inf) & ~deleted_mask)
	•	Apply denoising update for alive residues
	•	Sample split events at step boundary: for nodes with remaining unborn children, sample Bernoulli(1 - exp(-λ_split * Δt)) where λ_split = clamp(split_remaining_hat, 0, remaining) * h(t), h(t)=1/(1-t)
	•	If split fires: set birth_time[child] = t_{k+1}, initialize as split-copy
	•	Sample deletion events at step boundary: for to_be_deleted nodes, sample Bernoulli(1 - exp(-h_del(t) * Δt)) where h_del(t)=1/(1-t)
	•	If deletion fires: set deleted_mask[i] = True (drops from attention immediately)
	5.	At t=1, final output:
	•	drop residues with deleted_mask = True (output formatting only, most already deleted)

Note: Tree topology is sampled upfront, but birth_time values are realized via split events during trajectory. Deletions are fully event-driven with fixed hazard (no learned deletion head).

⸻

Implementation plan (Python)

1) Add new data augmentation utilities

Create a new module (e.g. tree_coupling.py) with:
	•	augment_with_dummy_deletions(features, D, policy="duplicate_adjacent")
	•	Implements the "duplicate and insert before/after" policy for D dummy residues.  ￼
	•	Produces alignment length A=N+D, plus to_be_deleted flag.
	•	sample_motifs(features, strategy_cfg) -> motif_mask
	•	Reuse your existing motif selection strategies.
	•	build_gapwise_parent_and_birth(motif_mask, cfg) -> parent_idx, birth_time, topo_order
	•	For each gap (contiguous scaffold region in alignment):
	•	sample a random local binary tree (non-midpoint-biased)
	•	assign parent pointers (neighbor-only)
	•	sample birth times with Beta schedule, enforce causality
	•	Return concatenated arrays over A.

2) Define new batch props for tree coupling

Add a small enum (or constants) local to the new interpolant:
	•	TREE_PARENT, TREE_BIRTH, TREE_ALIVE, TREE_TO_BE_DELETED, TREE_CHILD1, TREE_CHILD2

3) Implement TreeInterpolant (new, simpler interpolant)

New file: tree_interpolant.py

Key methods:
	•	corrupt_batch(batch: BatchFeatures, task: DataTask) -> NoisyFeatures
	1.	(Optionally) augment with dummy residues to form alignment length A
	2.	sample motifs and set motif conditioning masks
	3.	sample parent_idx, birth_time, topo_order
	4.	sample t (and per-domain times if needed)
	5.	run creation-state DP in topo order to compute C_trans, C_rot (and optionally C_aa)
	6.	sample X_t for alive residues using Brownian / IGSO(3) bridges to X1
	7.	compute targets:
	•	remaining_direct_children(t) for split loss (count ∈ {0,1,2} for binary tree)
	•	no deletion targets (deletion is event-driven, not learned)
	8.	return noisy batch with:
	•	existing _t fields (trans_t, rotmats_t, aatypes_t, time scalars)
	•	tree arrays (parent_idx, birth_time, to_be_deleted, child indices) + alive mask
	•	supervision targets (remaining children count for split loss)
	•	sample(...)
	•	forward Euler sampler with scheduled births, masked updates, and event-driven deletion.

4) Model interface changes (minimal)

Add one head to your existing model (or reuse existing output dict):
	•	pred_split_remaining_hat : (B, A) or (B, A, 3) - predicted remaining children count

This uses x1-prediction semantics from Branching Flows: predict expected number of children remaining by t=1.

NO deletion head: deletion is event-driven via to_be_deleted flag + fixed hazard h_del(t)=1/(1-t).

5) Losses

In your training step:
	•	base losses (trans, rot, seq): masked to alive_mask & ~to_be_deleted & ~motif_mask
	•	split remaining loss: cross-entropy over {0,1,2} or L2 on remaining_direct_children_true(t)
		- masked to alive_mask & ~motif_mask
		- target: count of DIRECT children with birth_time > t
	•	NO deletion loss: deletion is event-driven via to_be_deleted flag + fixed hazard h_del(t)=1/(1-t)
		- no learned deletion head during training

6) Debug helpers
	•	parent_to_children(parent_idx) -> List[List[int]]
	•	validate_tree(parent_idx, birth_time, motif_mask):
	•	acyclic,
	•	causality satisfied,
	•	locality constraints per gap.

⸻

Notes for the implementer (important edge cases)
	•	Multiple gaps / multimers: parents must be constrained within each gap (and chain, if applicable).
	•	Masking (training): attn_mask = alive_mask = (t >= birth_time); deleted_mask NOT used during training
	•	Masking (sampling): attn_mask = alive_mask = (birth_time < inf) & ~deleted_mask; deleted_mask set by deletion events
	•	Loss masking: base losses use alive_mask & ~to_be_deleted & ~motif_mask
	•	Stochasticity: corruption DP is stochastic; you recompute it each corruption draw. The DP is linear in A and typically fine.
	•	Start simple: implement CA-only translations first (no rotations), confirm training runs, then add IGSO(3) rotations.


-----

ADDENDUM

1) Tree construction: interval recursion vs inheritance parent

There are two separate objects:
	•	(A) Interval-splitting recursion: an algorithm to decide which residue is created next inside an interval between two already-alive “boundary” residues.
	•	(B) parent_idx inheritance graph: the actual parent used for “split-copy” initialization at birth.

To avoid ambiguity, the recommended construction is:

Forward gap-growth algorithm (explicitly enforces “neighbor-only at birth”)

For each scaffold gap with final indices i0..i(L-1) lying between two motif boundary residues Lmotif and Rmotif (these are alignment indices in 0..A-1 and are alive from t=0):

Maintain a set of active intervals. Each interval is:
	•	left boundary index L (alive residue index)
	•	right boundary index R (alive residue index)
	•	a list of final scaffold indices I = [ ... ] strictly between them that are not yet born

Repeat until all scaffold indices are born:
	1.	Pick an active interval (L, R, I).
	2.	Choose a split index j ∈ I (random, not deterministic midpoint).
	3.	Choose inheritance parent as one of the boundaries: parent_idx[j] ∈ {L, R} (e.g., nearest boundary, or random with bias).
	4.	Assign birth_time[j] (see §4).
	5.	Mark j alive; replace interval (L, R, I) with up to two new intervals:
	•	(L, j, I_left) where I_left are indices < j
	•	(j, R, I_right) where I_right are indices > j

Key point: At the moment j is born, its chosen parent is always adjacent in the alive ordering (because it is one of its two alive boundaries). This is the intended meaning of “neighbor-only splits” (neighbor in the current alive list, not necessarily adjacent in final sequence indices).

This construction makes (A) and (B) consistent: the recursion defines the subinterval boundaries, and the inheritance parent is chosen from those boundaries.

Concrete example: gap [0,1,2,3,4] between motif boundaries

Let alignment indices be:
	•	left motif boundary L = 5
	•	right motif boundary R = 10
	•	scaffold positions in between are [0,1,2,3,4] (for illustration; in practice these are the actual alignment indices in that region)

One valid non-midpoint tree / schedule:
	•	Step 1: interval (5,10), choose j=1, set parent[1]=5, birth[1]=0.20
	•	intervals become: (5,1,[0]) and (1,10,[2,3,4])
	•	Step 2: interval (5,1,[0]), choose j=0, set parent[0]=1, birth[0]=0.28
	•	interval becomes none (empty)
	•	Step 3: interval (1,10,[2,3,4]), choose j=4, set parent[4]=10, birth[4]=0.40
	•	intervals become: (1,4,[2,3]) and (4,10,[])
	•	Step 4: interval (1,4,[2,3]), choose j=2, set parent[2]=1, birth[2]=0.55
	•	intervals become: (1,2,[]) and (2,4,[3])
	•	Step 5: interval (2,4,[3]), choose j=3, set parent[3]=4, birth[3]=0.70

So for scaffold indices [0,1,2,3,4]:
	•	parent_idx = [1, 5, 1, 4, 10]
	•	birth_time = [0.28, 0.20, 0.55, 0.70, 0.40]

Motif roots: parent[5]=5, birth[5]=0; parent[10]=10, birth[10]=0.

This is a binary tree over the gap, but represented purely by arrays.

⸻

2) Wording: “non-midpoint-biased” vs “biased away from midpoint”

Replace with: “non-deterministic split-point selection (not always midpoint)”.

You can optionally add a tunable bias (toward ends, toward center, etc.), but the main goal is: don’t always pick the midpoint.

⸻

3) Discrete sequence handling: make the schedule explicit

Use a masked-token bridge that depends on the residue’s “local time” since birth.

Define local progress:
u = \mathrm{clip}\left(\frac{t - b_i}{1 - b_i},\; 0,\; 1\right).

Define an unmask probability schedule (example):
p_\text{keep}(u) = u^\gamma
with \gamma \in (0,2] (tunable).

Then for an alive residue:
	•	with probability p_\text{keep}(u): set aatypes_t[i] = aatypes_1[i]
	•	else: set aatypes_t[i] = MASK

This is the simplest “bridge-like” discrete corruption: it starts fully masked at birth (u=0) and becomes fully revealed at t=1 (u=1). If you want a richer corruption, you can replace the “MASK” branch with a D3PM-style categorical noising kernel, but the above is sufficient to start and matches common masked diffusion practice.

Creation-time token (C_aa[i]) is therefore always MASK; the bridge is defined by this local schedule, not by Brownian math.

⸻

4) Birth-time causality cascade: avoid “push to 1”

Agreed. Don’t sample all birth times independently and then “max() with parent + ε”.

Instead, sample in topological (birth) order conditioned on the parent:

For each born scaffold node i with parent p:
b_i = b_p + (1 - b_p)\,u,\quad u \sim \mathrm{Beta}(\alpha,\beta)

This guarantees b_i > b_p without any cascade, and gives you direct control:
	•	α<β makes births early (more of the scaffold appears sooner)
	•	α>β makes births late
	•	you can also add depth-dependent bias if desired

No global ε is needed except a tiny clamp for numerical stability.

⸻

5) “Zero variance at birth” (t = s)

Correct: for Brownian bridges, variance at exactly t = birth_time is zero, so a residue is born exactly at its parent’s sampled location at that time. This is intended for a split-copy birth rule.

Two clarifications:
	•	This does not require the model to “predict endpoints.” During training, endpoints are the data (X1); the model predicts a denoising target / vector field / score at time t, as in normal flow matching. The fact that new residues start coincident is not inherently harder; it’s just a choice of coupling.
	•	If you want nonzero stochasticity at creation, add birth jitter:
	•	translations: C_trans[i] = x_parent(b_i) + Normal(0, σ_birth^2 I)
	•	rotations: C_rot[i] = R_parent(b_i) * IGSO3(τ_birth)
with small σ_birth / τ_birth. This keeps the semantics but avoids exact overlap. Optional.

⸻

6) Motif boundary / roots

Yes: motif residues are the roots (alive at t=0). For each scaffold gap, the two motif boundary residues are the initial alive boundaries that seed the gap-growth process.

Scaffold residues do not form an independent tree disconnected from motifs unless you explicitly want that. In the recommended scheme:
	•	the first scaffold born in a gap copies from one of the motif boundaries,
	•	later residues copy from subinterval boundaries (which may be motifs or previously born scaffolds).

So the forest is rooted in motifs.

⸻

7) Split prediction at inference: event-driven sampling (chosen design)

**Chosen design: Event-driven split sampling (Branching Flows style)**

The split head drives event sampling online during inference:
	•	Model outputs split_remaining_hat[i](t) - predicted remaining children count (x1-prediction)
	•	Effective split rate: λ_split_eff = clamp(split_remaining_hat[i], 0, remaining_direct_children) * h(t)
	•	Where h(t) = 1/(1-t) ensures all births complete by t=1
	•	Sample split events as Bernoulli(1 - exp(-λ_split_eff * Δt)) at step boundaries
	•	Tree topology (parent_idx, child indices) is sampled upfront, but birth times are realized via split events during trajectory

This is the Branching-Flows-faithful approach with learned hazard control through the x1-prediction head.

⸻

8) To-be-deleted residue handling at inference (chosen design)

**Chosen design: Inference includes to-be-deleted nodes with event-driven deletion**

During sampling (generation):
	•	Sample tree topology including ~20% extra to-be-deleted leaves (same as training augmentation proportion)
	•	These nodes are attached to the tree like regular scaffold nodes (via parent_idx, child indices)
	•	They are initialized as split-copies when their parent spawns them (birth_time set by split events)
	•	Deletion is purely event-driven: no endpoints needed, just the to_be_deleted flag + fixed hazard h_del(t)=1/(1-t)
	•	Deletion events sample Bernoulli(1 - exp(-h_del(t) * Δt)) at step boundaries for to_be_deleted nodes
	•	When deletion fires: deleted_mask[i] = True (drops from attention immediately)
	•	At t=1: drop all positions with deleted_mask=True (output formatting; most already deleted by exploding hazard)

This provides the same length variability mechanism during inference as in training, and tests the model's ability to handle dynamic length changes during the trajectory.

⸻

9) Loss masking for to-be-deleted residues

**Recommended masking:**
	•	Base denoising losses (trans, rot, seq): mask with
		loss_mask = alive_mask & ~to_be_deleted & ~motif_mask
	•	Split remaining loss: mask with alive_mask & ~motif_mask
		- Target: remaining_direct_children_true(t) ∈ {0,1,2}
	•	NO deletion loss: deletion is event-driven via to_be_deleted flag + fixed h_del(t)=1/(1-t)
		- No learned deletion head during training

Exclude to-be-deleted residues from base losses so the model isn't encouraged to "model the geometry" of dummy augmentation positions that will be removed. Also exclude motifs from base losses (fixed conditioning).

⸻

10) Batched DP efficiency (creation-state pass)

You’re right: per-example topological DP is sequential. Practical options:

Option 1: depth-bucketed vectorization (recommended)

Compute depth[i] = depth[parent[i]] + 1 (motifs depth 0). Then process d = 0..max_depth:
	•	gather indices with depth == d
	•	for those indices, compute parent state at birth[i] and set creation state

This is GPU-friendly because each depth layer is parallel. With the forward gap-growth tree, depth is usually O(\log L) for balanced-ish splits, but can be larger; you can cap depth by construction (e.g., avoid degenerate splitting).

Option 2: discretize birth times into K bins

Assign each node to a bin by birth time and process bins in order (parents always in earlier bin). This gives a fixed K loop.

Option 3: accept a small Python loop

If A is ~1–2k and batch sizes are modest, a compiled torch.compile loop over A can be acceptable for an MVP. You can optimize later.

⸻

11) Model input: does it see the tree?

You do not need to feed parent_idx to the model.

**Model inputs:**
	•	Noisy state: trans_t, rotmats_t, aatypes_t
	•	Time: t (and possibly local progress u or age = t - birth_time)
	•	Masks: alive_mask, motif_mask, diffuse_mask
	•	Optional tree features: birth_time (for age embedding)

**Model outputs:**
	•	Denoising predictions: trans, rot, seq
	•	split_remaining_hat: predicted remaining children count (x1-prediction)
	•	NO deletion output: deletion is event-driven via to_be_deleted flag + fixed h_del(t)=1/(1-t)

The split head is a function of the current noisy state + time + conditioning, without explicit parent pointers. The parent pointers are primarily a corruption/supervision device, not a model input.

If you later move to Mode B (learned hazards), you still don't strictly need parent_idx; you need a per-residue signal to score where growth should occur (gap id, proximity to motifs, etc.), which can be learned from state and masks.


-----

Addendum: clarify indexing, topo_order, and split-point default

1) Example indexing (use consistent alignment indices)
Use actual alignment indices throughout. Example:
	•	Alignment indices 0..14
	•	Motif residues: 0..4 and 10..14 (alive at t=0)
	•	Scaffold gap residues: 5..9 (5 residues between the motifs)

So boundaries for the gap are L=4, R=10, and scaffold set is I = [5,6,7,8,9].

One valid growth / parent assignment:
	1.	Interval (4,10, [5,6,7,8,9]), pick j=6, set parent[6]=4, birth[6]=0.20
New intervals: (4,6,[5]), (6,10,[7,8,9])
	2.	Interval (4,6,[5]), pick j=5, set parent[5]=6, birth[5]=0.28
New intervals: none
	3.	Interval (6,10,[7,8,9]), pick j=9, set parent[9]=10, birth[9]=0.40
New intervals: (6,9,[7,8]), (9,10,[])
	4.	Interval (6,9,[7,8]), pick j=7, set parent[7]=6, birth[7]=0.55
New intervals: (6,7,[]), (7,9,[8])
	5.	Interval (7,9,[8]), pick j=8, set parent[8]=9, birth[8]=0.70

Result (for indices 5..9):
	•	parent_idx[5]=6, parent_idx[6]=4, parent_idx[7]=6, parent_idx[8]=9, parent_idx[9]=10
	•	birth_time[5]=0.28, birth_time[6]=0.20, birth_time[7]=0.55, birth_time[8]=0.70, birth_time[9]=0.40
Motifs: parent[i]=i, birth[i]=0 for i ∈ {0..4,10..14}.

This keeps one numbering scheme: alignment indices only.

2) topo_order is implicit in construction
Yes. If you assign birth_time at the moment you choose j during gap-growth, then the construction order is already a valid topological order (parents/boundaries exist before children by construction).

Spec update: define
	•	topo_order = [j_0, j_1, ..., j_{S-1}] = the sequence of scaffold indices selected during interval growth (optionally preceded by motif indices if you want a full-length order).
You can still compute argsort(birth_time) as a fallback sanity check, but you typically don’t need to store both.

3) Split-point distribution: pick a concrete default
Recommended default for MVP:
	•	Default: j ~ Uniform(I) (uniform over remaining indices in the chosen interval)

Optional knobs (explicitly documented but off by default):
	•	End-biased (“stringy” growth): pick j from the first/last k indices of I with probability p_end, else uniform
	•	Beta-position: sample u ~ Beta(a,b), map to a rank r = floor(u * |I|), choose j = I[r]

But the spec should state: default uniform; bias is an experimental option.
