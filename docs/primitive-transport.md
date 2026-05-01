# Primitive-Native Transport

`CanonicalPathPlan` remains the single internal source of truth.

The executor transport now uses `PrimitivePathPlan` only.

Current transport rules:

- web/API contracts stay unchanged
- preview remains derived from canonical sampling
- commit publishes:
  - `/wall_climber/primitive_path_plan`
- the executor consumes `PrimitivePathPlan`
- `DrawPlan` is no longer published or accepted as runtime transport

Decommission rule:

- `DrawPlan` may still appear in diagnostic/export helpers while older tooling is being
  cleaned up, but it is no longer part of the live execution path
