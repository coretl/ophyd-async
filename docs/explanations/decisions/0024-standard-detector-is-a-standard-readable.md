# 24. StandardDetector is a StandardReadable

Date: 2026-10-30

## Status

Accepted

Partly superseded by [ADR 0027](0027-areadetector-statistic-names.md), under which
`AreaDetector` registers every plugin.

## Context

[ADR 0012](0012-detector-rewrite.md) gave `StandardDetector` its own ways to put a signal
in a step scan: `add_config_signals` for configuration, and the `prepare_single` data
logic tier with `ReadableDataProvider` for values read once per event. Every other Device
does both by registering the signal on a `StandardReadable`, and
[ADR 0023](0023-runtime-readable-format.md) made that registry keyed by Device and
changeable at runtime, which is the per-plugin opt-in the #1395 reviewers asked for.

## Decision

### `StandardDetector` inherits `StandardReadable`

A detector's step-scan signals are registered like any other Device's. The detector does not
override `read()`, `describe()` or `hints`: each verb gathers a tuple of callables and the
detector appends its own, as `StandardFlyable` does to `_stage_funcs`, because overriding
meant re-deriving the registered children's contributions in each verb, which would drift.
`add_config_signals` becomes a deprecated wrapper.

A trigger logic computes deadtime from all the detector's configuration signals rather than
nominating its own, as the values that determine deadtime ought to be recorded anyway. They
come from the registry, not `read_configuration()`, so read-only signals such as
`ADBaseIO.model` count.

`AreaDetector` registers its driver but no plugin: a plugin is opted in with
`set_readable_format(plugin, StandardReadableFormat.CHILD)` and out with `None`, the
per-plugin, reversible choice the #1395 reviewers asked for.

### The `prepare_single` data logic tier goes

Registering a signal now gets one value per event into `read()` for any Device, which was
all [ADR 0012](0012-detector-rewrite.md)'s `prepare_single` tier did. It is deleted with its
providers rather than kept as a second way to do one thing, the one that drops itself with a
warning once a scan asks for more than one collection.

## Consequences

- A detector's signals can be switched between techniques, stored and restored like any
  other Device's ([ADR 0023](0023-runtime-readable-format.md)).
- A signal that determines deadtime must be registered as configuration.
- Out-of-tree `prepare_single` data logics must register their signals instead; there is no
  shim.
