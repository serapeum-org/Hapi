# Protocols

`Run` does not import `Catchment`, and `Catchment` does not inherit from anything in the run
layer. What connects them is stated here: the run layer owns the interfaces, and `Catchment`
satisfies them structurally.

That is dependency inversion, and it buys two checkable things. `hapi.run` and `hapi.wrapper`
carry no runtime dependency on the concrete class, so the arrow between the modules points the
other way. And the requirement is checked by mypy, where it used to live in prose in each
method's docstring — prose does not fail CI.

`CatchmentLike` is deliberately builder-shaped: its fields really are optional, because a
half-built catchment is a legitimate state. Narrowing it into a run
(see [Runs](runs.md)) is where the optionality is resolved.

## CatchmentLike
::: hapi.protocols.CatchmentLike

## SupportsQsim
::: hapi.protocols.SupportsQsim

## SpatialDistribution
::: hapi.protocols.SpatialDistribution
