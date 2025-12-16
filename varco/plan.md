Variable Cogeneration (Varco) Implementation Plan (Branching Flows for proteins)

0. Goals and scope

Goal: Extend your existing protein FM model (translations + rotations + sequence) to support variable-length generation via split (insertion) + deletion events, for:
	1.	Unconditional generation (sample full chains).
	2.	Motif scaffolding (motif residues fixed; model generates variable-length scaffold segments between/around motifs).

Non-goals (v2):
	•	Side-chain atom generation.
	•	Perfect faithfulness to “tree coupler heuristics” (we want correctness of semantics + stable engineering first).
	•	We follow the paper’s split/deletion semantics (R_t and rho_t), not the older {0,1,2} “direct children count” heuristic.

⸻

1. High-level architecture

Keep the same separation of concerns you already started (tree coupling, interpolant/corruptor, model heads, sampler)  ￼, but change the semantics of the branching pieces to match Branching Flows:

Core components
	•	TreeCoupling: samples latent Z that fully specifies how X_0 maps to X_1 via splits + deletions.
	•	TreeInterpolant: draws corrupted states X_t \sim p(\cdot \mid Z) using your base bridges (R³ + SO(3) + discrete).
	•	VarcoFlowModel: predicts
		•	base end-state targets (\hat x_1 for translations/rotations/AA logits),
		•	split head \hat R_t ≥ 0: predicted remaining number of split events by t=1 (counting-flow “remaining increments”),
		•	deletion head \hat\rho_t ∈ [0,1]: predicted probability this element will be deleted by t=1 (deletion flow).
	•	Sampler: simulates marginal generation using the model’s base generator + split/deletion rates.
		Note: Z/tree is used to DEFINE training-time conditional corruptions and targets; inference samples the marginal branching process and does not require sampling a tree.

This mirrors the paper’s “learn base generator + split and deletion rates” framing (their protein finetune adds both branching and deletion and a conditioning mask)  ￼.

---

1a. Organization

This document outlines the implementation plan for varco, building on top of the cogeneration codebase. The goal is to reuse as much existing infrastructure as possible while adding the new tree-based variable-length generation capability.

```
varco/
├── spec.md                     # Specification document
├── implementation_plan.md      # This file
├── config.py                   # VarcoConfig dataclass (extends cogeneration config)
├── tree_coupling.py            # Tree construction, birth times, parent assignment
├── bridges.py                  # Brownian bridge (R3) and IGSO3 bridge samplers
├── interpolant.py              # TreeInterpolant class
├── batch_props.py              # TreeBatchProp enum for new batch keys
├── model.py                    # VarcoFlowModel with base + split(R_t) + deletion(rho_t) heads
├── node_features.py            # Extended node feature net (birth_time, age embeddings)
├── heads.py                    # Split head (predict remaining splits R_t) + deletion head (predict rho_t)
├── losses.py                   # base loss + split(R_t) loss + deletion(rho_t) loss
├── module.py                   # VarcoModule (Lightning)
└── sampler.py                  # Forward sampling with model-predicted split/deletion rates (no tree at inference)
```

1b. What We Reuse from Cogeneration

### Fully Reusable (import directly)
| Module | Usage |
|--------|-------|
| `dataset/datasets.py::BaseDataset` | Data loading, filtering, train/test splits |
| `dataset/featurizer.py::BatchFeaturizer` | ProcessedFile -> BatchFeatures |
| `dataset/motif_factory.py::MotifFactory` | Generate motif segments from mask |
| `data/so3_utils.py` | SO3 utilities, IGSO3 sampler table |
| `data/rigid.py`, `data/rigid_utils.py` | Rigid/frame utilities |
| `data/const.py` | Constants (MASK_TOKEN_INDEX, etc.) |
| `data/noise_mask.py` | Noise generation utilities |
| `data/all_atom.py` | Frame to atom conversion |
| `models/attention/*` | IPA trunk, attention modules |
| `models/edge_feature_net.py` | Edge feature embedding |
| `models/aa_pred.py` | Amino acid prediction network |
| `models/embed.py` | Time/position embeddings |
| `models/confidence.py` | pLDDT, PAE modules (optional) |
| `type/batch.py::BatchProp`, `NoisyBatchProp` | Base batch property enums |
| `type/task.py::DataTask` | Task enum |
| `config/base.py::BaseClassConfig` | Config base class |
| `data/trajectory.py` | SamplingTrajectory for inference |

### Partially Reusable (extend or reference)
| Module | What to Reuse | What to Change |
|--------|---------------|----------------|
| `data/interpolant.py::Interpolant` | Interface pattern (`corrupt_batch`, `sample`) | New TreeInterpolant with bridge-based corruption |
| `data/fm/translations.py` | Prior sampling, vector field formula | Add Brownian bridge sampling |
| `data/fm/rotations.py` | IGSO3 sampler, geodesic utilities | Add IGSO3 bridge sampling |
| `models/model.py::FlowModel` | Architecture pattern, recycling | Add split head (R_t) + deletion head (rho_t), plus birth_time/age inputs |
| `models/node_feature_net.py` | Embedding logic | Add birth_time/age embedding |
| `models/module.py::FlowModule` | Training loop pattern | New VarcoModule with tree losses |
| `models/loss_calculator.py` | Loss aggregation pattern | Add split(R_t) loss + deletion(rho_t) loss |
| `config/base.py` | Config structure | New VarcoConfig |

### Not Reused (new implementation)
| Component | Reason |
|-----------|--------|
| Tree construction (parent_idx, birth_time, to_be_deleted) | Core varco feature |
| Creation-state DP | Core varco feature |
| Bridge samplers | Different from standard flow matching |
| Split head + targets (R_t): new supervision target derived from subtree leaf counts. | New branching supervision |
| Deletion modeling (rho_t + deletion_time): simulate deletion in training corruption and in inference sampling. | New deletion semantics |
| Variable-length sampling: model-predicted split/deletion event rates; maintains insertion order without sampling a tree. | New event-driven sampling |


⸻

2. Representations and invariants

2.1 Fixed-size tensor representation

Use a fixed maximum capacity A = A_max nodes per example (per chain or per whole complex) with masks:
	•	alive_mask(t): node exists at time t (born and not deleted).
	•	deleted_mask(t): node has been deleted by time t (dynamic state in sampling; can be simulated in training).
	•	conditioning_mask: fixed residues (motifs / fixed chains) that do not update (see §6).

2.2 Stable node identity and ordering

Each node has immutable bookkeeping:
	•	node_id (0..A-1), parent_id, segment_id (which designable segment / chain), and a stable insertion order key (e.g., depth-first index in the sampled tree).
	•	Avoid “birth-order children arrays” as a first-class concept; store children as left/right (structural order) to prevent sampler logic from depending on step-size.

2.3 Batch properties

Tree structure (not exposed to model -- may just be data structure)
	•	tree_parent_idx (int64, -1 for root)
	•	tree_child_left_idx, tree_child_right_idx (int64, -1 if none)
	•	tree_segment_id (int64) – which scaffold segment / chain group
	•	tree_topo_order (int64) – cached parent-before-child order for DP corruption
	•	tree_subtree_leaf_count (int64) – terminal leaf count per node (defines R_true)
	•	tree_leaf_order (int64) – stable final order key for X1_plus leaves / training targets (training-only)

Store children in structural left/right order, not “birth order”, to avoid step-size dependent logic.

Times / labels
	•	tree_birth_time (float32 in [0,1])
	•	tree_leaf_is_deleted (bool) – leaf is “to-be-deleted” in augmented X1+ø
	•	tree_deletion_time (float32 in [0,1] or +inf) – only for leaves marked deleted

Derived Masks
	•	alive_mask(t) = (t >= birth_time) & ~deleted_mask(t)
	•	conditioning_mask (already in your batches; motifs/fixed residues)

⸻

3. TreeCoupling: sampling the latent Z

3.1 What Z contains

Sample a latent package Z that makes the conditional process Markov:
	•	A binary forest encoded as arrays over alignment length A (parent_idx, child_left/right_idx).
	•	Per-node birth_time in [0,1], with parent born earlier (causality).
	•	Per-leaf to-be-deleted label (from X1+ø augmentation via duplication).
	•	Optional helper: topo_order (indices in a valid parent-before-child order).

3.2 Choosing X_0 sizes

Match the Branching Flows protein setup conceptually:
	•	Unconditional chains: initialize each chain with a small random number of roots, e.g. 1 + Poisson(λ) (they used this kind of scheme)  ￼.
	•	Motif-scaffold segments: initialize each designable segment with one root element (their BranchSegment does this)  ￼.

3.3 “To-be-deleted” augmentation

Support an augmentation factor dr > 1 for leaf counts (duplicate some real leaves to create realistic deletable leaves). These duplicated leaves define which branches are destined for deletion; deletions are realized as actual removal events both during training corruption (via deletion_time/deleted_mask) and during inference sampling (via h_del(t) * \hat\rho_t).

3.4 Gap-wise tree construction (neighbor-only at birth)

3.4.1 Grouping
• Define groups = (chain_id, segment_id). Each group is one contiguous designable segment (motif scaffolding) or a whole chain (unconditional).

3.4.2 X1+ø augmentation (to-be-deleted leaves)
• For each group, create X1_plus by duplicating and inserting D extra residues adjacent to real residues (optionally mask AA state for dummies).
• Mark inserted duplicates with tree_leaf_is_deleted=True (to-be-deleted).

3.4.3 Forward gap-growth tree (interval recursion)
• For each designable gap bounded by two always-alive boundary residues (motif boundaries), maintain active intervals (L, R, I) where L and R are alive boundaries and I is the list of not-yet-born indices strictly between them in final order.
• Repeat until I is empty across all intervals:
  1) pick an active interval (L, R, I)
  2) choose a new index j ∈ I (default uniform; optional bias is a tuning knob)
  3) set parent_idx[j] ∈ {L, R} (e.g., nearest boundary or random; this enforces neighbor-only at birth)
  4) sample birth_time[j] conditioned on parent: b_j = b_parent + (1 - b_parent) * u, u~Beta(α,β)
  5) split the interval into (L, j, I_left) and (j, R, I_right)
• Optionally record child_left/right_idx from parent_idx (for supervision convenience).

3.4.4 topo_order
• Define topo_order as the order in which indices are created by the interval recursion (parents/boundaries always precede children).

3.5 Birth-time schedule (no causality cascade)
- Sample each child birth time from a split hazard distribution H_split on (b_parent, 1), to guarantee causality and completion by t=1.
- Implementation: b_i = H_split^{-1}(F(b_parent) + (1 - F(b_parent)) * u), u~Uniform(0,1), where F is the CDF of H_split.
- MVP default: choose h_split(t)=1/(1-t) (clamp near t=1 for stability).

⸻

4. Conditional corruption: sampling X_t \mid Z

4.1 Base bridges (reuse)

Reuse your existing corruptors/bridges, but apply them per alive node:
- Translations: Brownian / OU bridge from creation state C_trans[i] at birth_time[i] to endpoint trans_1[i].
- Rotations: IGSO(3) bridge from creation state C_rot[i] at birth_time[i] to endpoint rot_1[i].
- Sequence: discrete substitution bridge toward endpoint AA label using local progress u=(t-b_i)/(1-b_i) (see §4.4).


4.2 Handling splits and deletions in the conditional process
	•	Splits: at a split time, duplicate the parent's current element state into the child (in-place duplication).
	•	Deletions: realized during TRAINING corruption and inference sampling as actual removal events on branches that end in deleted leaves; we model this via deletion_time + deleted_mask(t) (paper's deletion operator).

4.2.1 Birth jitter
To avoid exact positional overlap at birth, optionally add small noise to the creation state:
	•	C_trans[i] = x_parent(b_i) + Normal(0, σ_birth² I)
	•	C_rot[i] = R_parent(b_i) · IGSO3(τ_birth)
with small σ_birth / τ_birth. This keeps split-copy semantics but prevents degenerate coincident positions. Default: off (exact duplication).

4.3 TreeInterpolant.corrupt_batch (creation-state DP + bridges)

1) Sample t ~ Uniform(0,1).
2) Compute a valid topo_order (from construction or argsort(birth_time) with parent-before-child tie-break).
3) Creation-state DP (single topological pass):
   • For roots/conditioned motif nodes: set creation state C[i] at birth_time=0 (usually equal to X1 state for motifs, or a chosen prior for unconditional roots).
   • For each non-root node i in topo_order:
     - let p = parent_idx[i], s = birth_time[i]
     - sample parent state at time s via a bridge from C[p] to X1[p]:
         trans_p(s) ~ Brownian/OU bridge(C_trans[p], trans_1[p], t=s; start=birth_time[p])
         rot_p(s)   ~ IGSO3 bridge(C_rot[p], rot_1[p], t=s; start=birth_time[p])
     - set child creation state as split-copy: C[i] = parent_state(s)
     - set creation AA token C_aa[i] by sampling from the background AA distribution (same as §4.4 when u=0).
4) Alive/deleted masks: alive[i] = (t >= birth_time[i]); deleted[i] = (t >= deletion_time[i]) for branches marked deleted in Z (else deleted=False).
5) For each i:
   • if not alive OR deleted: set state to padding (trans=0, rot=I, aa=PAD/UNK) and ensure attn/res masks exclude it.
   • if alive and not deleted: sample X_t[i] using a bridge from C[i] to X1[i] at time t (per modality).

4.4 Discrete sequence corruption (noisy jumps; no MASK token)
Following edit-flows style, sequence corruption uses noisy categorical jumps, NOT masking:
- The only "insertion" event is the birth itself (residue becomes alive).
- At birth (u=0), sample C_aa[i] from a background AA distribution (uniform or empirical frequencies over 20 amino acids).
- Use a D3PM-style transition matrix for a proper discrete bridge.
	- i.e. do not use simple mixture: u_i = clip((t - b_i)/(1 - b_i), 0, 1).
- For alive residues at time t, sample aatypes_t[i]: with probability u_i use endpoint aatypes_1[i]; with probability (1-u_i) sample from background.
- This means we always have a concrete amino acid, never MASK. The corruption is: (1) birth event and (2) noisy categorical jumps toward the endpoint.

4.6 Batched creation-state DP efficiency
- Default (MVP): a single topological loop over topo_order per example (cannot precompute much because Z changes per sample).
- Cache topo_order from TreeCoupling in the batch to avoid recomputation inside corrupt_batch.
- Future optimizations: vectorize across examples by processing batches of indices at the same depth, but correctness is defined by topo_order.

⸻

5. Model: trunk + heads

5.1 Trunk reuse

Reuse the existing IPA trunk / embeddings; extend node features with:
	•	t embedding
	•	alive_mask / conditioning_mask
	•	optional: segment_id embedding
	•	optional (debuggable): node_age = t - birth_time and/or depth features
	•	Note: parent_idx is NOT fed to the model; the tree is a corruption/supervision device (the model sees state + time + masks + birth_time/age).

5.2 Outputs (heads)

For each node i:
	1. Base end-state prediction \hat x_{1,i}: translation, rotation, AA logits.
	2. Split head \hat R_{t,i} ≥ 0: predicted remaining number of splits for this element by t=1.
	3. Deletion head \hat\rho_{t,i} ∈ [0,1]: predicted probability this element will be deleted by t=1.

⸻

6. Training losses and masking

6.1 Masks
	•	Conditioned residues (motifs): always present, always excluded from base losses, and prevented from updates during forward passes (matching BF-ChainStorm’s conditioning approach)  ￼.
	•	Designable residues: contribute to base losses.
	•	To-be-deleted leaves: excluded from base denoising losses but INCLUDED in split/deletion supervision (they define rho targets and drive deletion in corruption).

6.2 Loss terms

For each batch, sample t, sample Z, sample X_t | Z, then compute:
	1. Base FM loss (trans/rot/seq): masked to designable + alive + not-deleted-at-t.
	2. Split loss: supervise \hat R_t against the true remaining splits implied by Z at time t (counting-flow target).
	3. Deletion loss: supervise \hat\rho_t against whether the element/branch is destined to be deleted (binary).

6.3 Targets and losses

Base loss
• On nodes that are alive, not deleted at time t, and designable, supervise base prediction toward endpoint x1.
• Motif/fixed nodes excluded and frozen.

Split target (remaining splits)
• Let w_i be the number of terminal leaves in element i’s subtree (subtree_leaf_count).
• Paper conditional split intensity is h_split(t) * (w_i − 1); define R_true(i,t) = w_i − 1 (nonnegative integer).
• Train \hat R_t with a Poisson-like Bregman divergence (or MSE as MVP), treating \hat R_t as a nonnegative scalar.

Deletion target (deletion probability)
• rho_true(i,t) = 1 if i’s terminal leaf is marked deleted in Z, else 0.
• Train \hat\rho_t with binary cross-entropy.
6.4 Loss masking (explicit)
- Training corruption uses alive_mask(t) = (t >= birth_time) and deleted_mask(t) from sampled deletion_time for deleted leaves.
- Base losses: mask = alive_mask & ~deleted_mask & ~motif_mask & ~to_be_deleted.
- Split + deletion losses: mask = alive_mask & ~deleted_mask & ~motif_mask.

⸻

7. Sampling / inference

7.1 Initialization
	•	Build initial X_0:
	•	unconditional: sample root count per chain; sample root frames + AA from your base prior.
	•	motif scaffolding: fixed motifs inserted as conditioned nodes; each designable segment starts with one root.

7.2 Time stepping loop

At each step:
	1.	Run the model on current active nodes (alive & ~deleted).
	2.	Apply base update step (your chosen ODE/SDE solver) toward \hat x_1.
	3.	Sample split events with rate λ_split = h_split(t) * \hat R_t; on split allocate a free slot, duplicate parent state into child, set child alive.
	4.	Sample deletion events with rate λ_del = h_del(t) * \hat\rho_t; on delete mark deleted_mask=True.

7.2.1 Split / delete event sampling details

	•	Define hazard schedules h_split(t), h_del(t) (one place in the doc).
	•	Per active node:
	•	λ_split = h_split(t) * \hat R_t
	•	λ_del   = h_del(t) * \hat\rho_t
	•	Event sampling method: per-step Bernoulli approximation initially is fine.
	•	On split: allocate from a per-example free-list, duplicate state, set alive.
	•	On delete: set deleted_mask=True and drop from attention immediately.
	•	At t=1: gather alive & ~deleted ordered by the sampler-maintained insertion_order_key (stable order from split events).
7.4 Clarification: tree usage
	- Training: sample Z (tree + deletion labels) to define a conditional corruption Xt|Z and targets R_true, rho_true.
	- Inference: do NOT sample a tree; simulate the marginal branching process using \hat x_1, \hat R_t, and \hat\rho_t with split/delete operators.

7.2.2 Hazard schedules (defaults)
- Default: h_split(t)=1/(1-t) and h_del(t)=1/(1-t), clamped with ε for numerical stability near t=1.
- Interpretation: exploding hazard encourages events to complete by t=1 under the per-step Bernoulli approximation.

7.3 Finalization
	•	Gather nodes with alive & ~deleted at t=1, ordered by the sampler-maintained insertion_order_key.
	•	Convert frames to atoms as usual.

⸻

8. Motif scaffolding specifics

8.1 Segment definition

Represent the design problem as a set of designable segments between fixed motif blocks (and optional fixed partner chains). Each segment has:
	•	its own initial roots (usually 1),
	•	its own tree coupling,
	•	shared global attention across the whole complex (motifs provide context).

8.2 Conditioning implementation

Implement conditioning exactly as “mask embedding + prevent frame updates” (the Branching Flows protein model explicitly does this)  ￼.

⸻

9. Testing and milestones

9.1 Unit tests (must-have)
	•	TreeCoupling invariants: leaf counts, deletion labels, stable ordering.
	•	Corruption invariants: alive_mask monotonicity, split correctness, creation-state DP consistency (at t=birth_time[i], child creation state equals parent sampled state).
	•	Sampler invariants: no “reviving” deleted nodes; capacity never exceeded.
	•	Tree validation: acyclicity, causality (birth_time[parent] < birth_time[child]), and max-children<=2.

9.2 Milestones
	1.	MVP-A (splits only): translations only; train split head R_t with counting-flow target.
	2.	MVP-B (add deletions): add deletion head rho_t; simulate deletion in corruption; include deletion loss.
	3.	MVP-C (full multimodal): add rotations + AA substitutions.
	4.	MVP-D (motif scaffolding): conditioning mask + freeze updates, per-segment initialization.

9.3 Edge cases

- Multiple gaps / multimers: enforce that parent selection and interval recursion stay within (chain_id, segment_id) groups.
- Masking rules: training uses alive_mask=(t>=birth_time); sampling uses alive_mask=(birth_time<inf) & ~deleted_mask.
- Newborn participation: newborn nodes are initialized at step boundaries and participate in the NEXT model forward pass (no extra forward per step).
- Output formatting: at t=1 drop deleted nodes and order by the sampler-maintained insertion_order_key (tree_leaf_order is training-only).

