# 23. Make readable format a runtime setting keyed by Device

Date: 2026-08-18

## Status

Accepted

## Context

In ophyd v1 a `Component`'s `kind` could be changed at runtime, so one Device could serve
several techniques: a monochromator's energy is configuration for one and a hinted read for
another. `StandardReadable` had no equivalent: `add_readables` appended bound methods into
tuples that did not record which device each came from, so nothing could be removed or
re-formatted afterwards (#1394).

In #1395 `StandardDetector` had its own `add_config_signals`, and reviewers would not have
areaDetector plugins registered automatically: a detector carries far more plugins than are
wired into its chain, so reporting all of them hides which values mean anything. Per-plugin
opt-in after construction is the primitive #1394 asks for.

Three designs were considered:

- **Chain parsing**, registering what the plugin graph wires up. Rejected as too large, and
  it fails once plugins cross IOCs, as every plugin would have to be exposed to ophyd.
- **One registry for formats and logic objects.** Rejected: a format is keyed by child,
  unbounded in number and merged by union, where a logic is keyed by role, one per role, and
  a conflict is an error, so the registry would branch on entry kind at every insert.
  Changing a format alters `describe()`; changing a logic, the protocols the Device
  satisfies.
- **A format registry keyed by device**, leaving logic objects alone. Chosen.

## Decision

### The registry is a `{device: format}` dict

`StandardReadable` maps each registered child to its format, in registration order, and
derives each verb's callables on demand, so the verbs always see the current registry.
`set_readable_format(device, format)` sets one entry, and `None` removes it.

- Keying by `Device` is identity keying, which a registry of children needs: `Device` has no
  `__eq__`, and `DeviceVector` and `DeviceMap` hash by `id()`.
- A tuple of pairs was tried for immutability, but rebuilding it per registration made
  construction O(n²): 77 ms for 2000 signals against 13 ms for the dict.
- It is a `cached_property`, because `AreaDetector` and `add_children_as_readables` register
  children *before* `super().__init__()`, which would discard anything `__init__` assigned.
- Only a `SignalR` takes a signal format, so the verbs get entries already typed.
- Registering a device twice replaces its format, as the old double contribution was an
  accident.
- The deprecated `ConfigSignal` and `HintedSignal` markers go: they compared equal to their
  target, which only worked while `add_readables` matched at registration.

### Runtime changes can be undone, and the baseline is sealed by a metaclass

@oliwenmandiamond pointed out that a runtime change makes the class definition misleading,
and two safeguards were proposed: class-declared formats immutable unless opted out, or a
separate `StandardDynamicReadable`. Both were rejected, as they block the case this ADR
exists for: #1394's monochromator *declares* `energy` as configuration, and locking it would
make a beamline get the class changed to switch technique. @jwlodek added that ophyd v1 has
always been mutable, NSLS-II codebases rely on it, and differences between the libraries are
a cost.

Adopted instead, from both: `StandardReadable.reset_readable_formats` puts a Device back
to what its class declared, so a plan can undo a retune without recording what it changed.
Its baseline is the class, so it discards a stored technique too.

The baseline is sealed by a metaclass. Formats are declared in annotations, by registration
before `super().__init__()` (all 24 in-tree Devices) and by registration after it (legal,
though unused). `type.__call__` is the only point after all three; a snapshot at the end of
`__init__` was measured to miss the third, so the reset would silently drop a declared
child. `Device` already has a metaclass, so this adds no machinery.

### Format changes apply between runs, not within one

A change part way through a run would make `describe()` disagree with the descriptor emitted
at its start. This is documented rather than enforced, as a Device cannot see run
boundaries.

### Formats are stored with settings, under a reserved key

A technique change wants values and formats together, so they share one file and one
`apply_settings`. Values stay flat, and one reserved key holds the format changes:

```yaml
energy: 7.0
<READABLE_FORMATS>:
  <ROOT_DEVICE>:
    energy: HINTED_SIGNAL
    temperature: null
```

- No version marker: a file without the key changes no formats.
- `<` cannot begin a Python identifier, so no attribute path collides with a reserved key.
  Unlike `*FORMATS*` it needs no YAML quoting, and `<ROOT_DEVICE>` avoids the `? '' :` form
  PyYAML emits for an empty key. `retrieve_settings` warns and drops an unknown `<...>` key,
  so a newer file loads in an older version.
- Formats are keyed by owner, then child, because a format belongs to a pair: one child can
  be registered on two owners with different formats. Recording the owner needs no "direct
  child" rule, so `driver.acquire_time` and `DeviceVector` elements work.
- Formats are stored even by a config-only `store_settings`, as they decide which signals
  are configuration.

### Only changes are stored, and applying resets to the class defaults first

`walk_readable_formats` stores how each `StandardReadable` differs from its construction,
and `apply_readable_formats` resets the tree to its class defaults, then applies that.
Rejected:

- **Store everything, apply by replacing**: an old file silently unregisters a signal added
  in a later version, and undoes a release's change of default.
- **Store everything, apply by merging**: cannot return from technique B to A, as what B
  changed and A does not mention stays.

Every path is resolved before anything changes, so a bad path cannot half apply.
`Settings.partition` copies the formats to both halves, because plans such as
`apply_panda_settings` apply only the halves, and a copy on one would let the plan decide
whether they are restored. Formats go after values, so a failed write leaves them untouched.

## Consequences

- A Device can switch, save and restore techniques at runtime, closing #1394.
- `add_readables` keeps working but gains "set" semantics, so it is deprecated in favour of
  `set_readable_format`, which says what it does.
- A runtime change to a device registered outside its owner has no stable path, so
  `walk_readable_formats` warns and skips it.
