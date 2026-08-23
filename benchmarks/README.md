# Benchmarks

Empty for now. This directory will hold direct measurements of performance for the
algorithms in `src/`, comparing the same algorithm across array backends.

The goal these benchmarks exist to verify: for every algorithm, the Metal array
path must run faster than the CPU array path once the problem size is large enough,
and the crossover size should be measured and reported, not guessed.
