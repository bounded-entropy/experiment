# Explicit sampling controls

`SamplingSpec` accepts optional `top_k`, `min_p`, `presence_penalty` and
`repetition_penalty` values. Explicit values enter scientific identity, cross
the resident wire and reach vLLM. When these controls are supplied, the sealed
turn records `sampling_parameters` read back from the constructed engine
parameters, including its defaults, seed and stop strings.

Omitting the new fields preserves the earlier canonical identity and wire
bytes. Per-call overrides and the native model template remain available
through the shared sampling interface.

These controls were introduced for frozen-model evaluation. Recording the
effective settings does not establish correct PPO/GRPO likelihood ratios:
vLLM's returned raw log probabilities need not include temperature, truncation
or history-dependent penalties. A policy-gradient experiment must qualify its
behavior/current-policy probability convention for its selected transformations
before using those probabilities as importance ratios. The new controls do not
perform that correction.
