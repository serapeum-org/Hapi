# Config

The schema of a YAML run configuration. Each block below is one top-level key of the file; the
rules tying them together — which blocks a given `spatial_resolution` requires, and which it
refuses — live on `RunConfig`.

Build a model from a file with
[`Catchment.from_yaml`](catchment.md#hapi.catchment.Catchment.from_yaml).

## RunConfig
::: hapi.config.RunConfig

## CatchmentConfig
::: hapi.config.CatchmentConfig

## MeteoConfig
::: hapi.config.MeteoConfig

## FlowNetworkConfig
::: hapi.config.FlowNetworkConfig

## ParametersConfig
::: hapi.config.ParametersConfig

## ConceptualModelConfig
::: hapi.config.ConceptualModelConfig

## GaugesConfig
::: hapi.config.GaugesConfig

## OutputsConfig
::: hapi.config.OutputsConfig
