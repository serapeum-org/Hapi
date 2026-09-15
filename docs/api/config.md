# Config

The schema of a YAML run configuration. Each block below is one top-level key of the file; the
rules tying them together — which blocks a given `spatial_resolution` requires, and which it
refuses — live on `RunConfig`.

Build a model from a file with
[`Catchment.from_yaml`](catchment.md#hapi.catchment.Catchment.from_yaml).

## RunConfig
::: hapi.core.config.RunConfig

## CatchmentConfig
::: hapi.core.config.CatchmentConfig

## MeteoConfig
::: hapi.core.config.MeteoConfig

## FlowNetworkConfig
::: hapi.core.config.FlowNetworkConfig

## ParametersConfig
::: hapi.core.config.ParametersConfig

## ConceptualModelConfig
::: hapi.core.config.ConceptualModelConfig

## GaugesConfig
::: hapi.core.config.GaugesConfig

## OutputsConfig
::: hapi.core.config.OutputsConfig
