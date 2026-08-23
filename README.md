# AlgorithmicNLA.jl

[![CI](https://github.com/jeremyiwk/AlgorithmicNLA.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/jeremyiwk/AlgorithmicNLA.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://jeremyiwk.github.io/AlgorithmicNLA.jl/dev/)
[![codecov](https://codecov.io/gh/jeremyiwk/AlgorithmicNLA.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jeremyiwk/AlgorithmicNLA.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![ColPrac: Contributor's Guide](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

The goal of ``AlgorithmicNLA.jl`` is to provide an interface for "abstract" numerical linear algebra algorithms. Following ``GenericLinearAlgebra.jl``, I want to create a repository that runs NLA across a variety of backends.

The ultimate goal is to fully abstract the backend. The proximate goal is to fill out the features currently lacking from ``Metal.jl``.

Supported Julia versions: 1.10 (current LTS) and later.

Contributions follow the [ColPrac](https://github.com/SciML/ColPrac) contributor guide.
