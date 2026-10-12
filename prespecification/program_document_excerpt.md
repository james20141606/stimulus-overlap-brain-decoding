# Program-document excerpt (design locked 2026-07-08; the static arm named here, a patch
# transformer, was replaced by the residual convolution after the MB-1 pilots of 2026-07-09;
# see the paper's Appendix I)

The hypothesis and decision gate of the architecture study, verbatim from the program
document as committed on 2026-07-08. The encoder pair was re-specified on 2026-07-09 after
three-subject pilots; the deciding 10-subject sweep was launched on 2026-07-26, and the
2026-07-30 notebook entry applies the gate to it as a paired bootstrap interval per budget.
That entry's interval [+2.05,+14.46] came from an earlier bootstrap run; the paper reports the
seeded 20,000-draw recomputation [+2.40,+14.19] by scripts/mb_conditions.py.

## The claim we are trying to break

**H1 (dynamic wins):** In LOW-SHOT cross-subject / cross-device transfer for
perceived-language decoding, a *dynamics-aware* token (how the latent neural state
MOVES) transfers better than a *static amplitude-patch* token, holding the
downstream head, targets, data, and calibration budget fixed.

**H0 (null we must try to confirm):** A well-tuned static patch matches or beats the
dynamic token at every calibration budget k. If H0 holds on both cross-subject and
cross-device, the program's core bet ("dynamic > static") is unsupported on this
data → we STOP before the RSN/clinical arms. Either outcome is publishable.

This is bet (a) of the two existential bets in the program (the other — does a
resting/clinical dynamics prior transfer to task dynamics — is deferred to MB-2,
which needs resting-state data).

## Kill-gate

If D ≤ S across all k with overlapping CIs on BOTH cross-subject AND cross-device →
H0 confirmed → write it up as a negative result; do NOT proceed to RSN/source view
or clinical arms. If D clearly beats S at small k on at least one axis with a clean
shuffled null → H1 supported → proceed to MB-2 (resting-prior transfer) and the
RSN-view ablation.
