# Architecture

SignDuplex V1.8 consumes only the causal RGB and Pose prefixes available at a decision boundary, together with the known system state. The unchanged backbone combines the MiniCPM-o visual tower, a Uni-Sign pose encoder, a learned causal Temporal Adapter, a Sign Projector, and attention LoRA in the language backbone.

The prediction layer is state-conditioned and factorized:

- `LISTENING` routes to a four-output head: `IDLE`, `INCOMPLETE`, `BACKCHANNEL`, `COMPLETE`.
- `SPEAKING` routes to a two-output head: `BARGE`, `NO_BARGE`.

This is a 4×2 routed design, not a unified six-class softmax. The V3.7.0 Interaction Controller consumes the raw routed prediction and applies the explicit state transition contract, Post-BARGE Protection, and Backchannel Cooldown. These controller policies do not modify the model logits.
