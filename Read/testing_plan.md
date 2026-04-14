# Testing And Next-Step TODO

## Current Status

- Standalone SoftGroup training on direct challenge `.npy` data now works end-to-end.
- The revised synthetic dataset plus SoftGroup default augmentation produced a very strong validation result.
- This means the current pipeline is already useful, but there are still clear data-generation and robustness questions worth addressing before treating the setup as final.

## Synthetic Data Generation TODO

- Revise color jitter so it explores hue more directly instead of only per-channel RGB gain/offset.
  The current jitter mostly changes brightness and color cast, not hue coverage.
- Make inserted-object point sampling less visibly artificial.
  Current sampling is spacing-aware voxel thinning with relaxation when too few points remain.
- Test removing or reducing spacing relaxation.
  This may preserve a more regular, grid-like point layout and make the inserted object look less noticeably synthetic.
- Revisit inserted-object density sampling policy.
  Compare current scene-matched spacing against more structured sampling that better mimics MultiScan surface sampling.
- Allow AABB contact rather than treating contact as collision.
  Current AABB logic rejects overlap and exact touching, which reduces valid stacked placements.
- Keep overlap rejection, but allow parent-child or general object contact when there is no true penetration.
  This should increase realistic stacking cases.
- Review stacking behavior after the AABB change.
  The generator still attempts stacking, but strict contact rejection likely reduces how many stacked placements survive.
- Clarify "increase placement probability" before changing it.
  Objects are already placed unless repeated attempts fail; the more meaningful knobs are:
  `MIN_INSERTIONS`, `MAX_INSERTIONS`, `STACK_ON_OBJECT_PROB`, placement-attempt budget, and rejection strictness.
- Consider increasing object count per scene only after placement realism is improved.
  More objects without better contact handling may mostly increase synthetic artifacts.

## Testing TODO

- Run more controlled ablations so improvements can be attributed cleanly.
  Important comparisons:
  current revised dataset with `aug_prob=0.0`
  current revised dataset with `aug_prob=1.0`
  old generator vs revised generator under the same training setup
- Add more adverse synthetic scenes:
  clustered inserted objects
  near-contact inserted objects
  many-object scenes
  tall stacks
  partially occluded placements
- Create a small stress-test subset specifically for difficult placement and separation cases.
- Track per-scene failure modes, not only aggregate F1.
  In particular, watch for:
  over-segmentation in crowded scenes
  missed instances in stacked scenes
  failures near contact boundaries
- Visualize more outputs routinely in `.ply`.
  Keep checking GT vs prediction for both strong scenes and failure scenes.

## Model And Feature TODO

- Test normals as an explicit model input.
  The data already preserves normals, but the current v1 SoftGroup path does not consume them.
- Compare current SoftGroup baseline against a normal-aware variant before moving to larger architecture changes.
- Consider SoftGroup++ later as a separate branch, not mixed into data-generation changes.
- Keep the current strong run as a stable baseline so future experiments are compared against a fixed reference.

## Evaluation TODO

- Keep challenge-style `F1@0.25` and `F1@0.50` as the primary metric.
- Keep native SoftGroup AP as a secondary diagnostic, not the main decision metric.
- Save the exact config, dataset root, and generation settings for every notable run.
  Recent gains were large enough that future comparisons should avoid mixing dataset and augmentation changes without noting both.

## Notes For Future Me

- The challenge normalization rule alone appears to distort the local metric scale that SoftGroup grouping expects.
- The current compromise of:
  challenge normalization
  plus fixed spacing restoration
  seems to work very well and should be treated as an important part of the current baseline.
- The recent large gain likely comes from a combination of:
  cleaner synthetic data
  stronger augmentation
  better compatibility between data scale and SoftGroup grouping
- Before making the model more complex, it is worth exhausting the data-engine improvements first.
