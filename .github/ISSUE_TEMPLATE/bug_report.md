---
name: Bug report
about: A crash, a wrong result, or an accuracy failure
---

**Describe the bug**

What happened, and what you expected instead. For accuracy issues, state the
element type, the problem size, and the observed versus expected error.

**Minimal reproducer**

```julia
# The smallest script that shows the problem.
```

**Environment**

Paste the output of:

```julia
using Pkg; Pkg.status("AlgorithmicNLA")
versioninfo()
```

If a GPU backend is involved, also paste the backend's device info (for Metal:
`using Metal; Metal.versioninfo()`).
