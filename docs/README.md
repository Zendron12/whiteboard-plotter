# Repository Documentation

This root `docs/` directory is a navigation layer for the repository.

Installable ROS package documentation lives under `src/wall_climber/docs/` so
`colcon build` can package it reproducibly without reaching outside the
`wall_climber` package directory. Those files are installed to
`share/wall_climber/docs`.

Package docs:

- [Canonical-first ingestion](../src/wall_climber/docs/canonical-first-ingestion.md)
- [Legacy compatibility](../src/wall_climber/docs/legacy-compatibility.md)
- [Primitive transport](../src/wall_climber/docs/primitive-transport.md)
- [X/CoreXY plotter foundation](../src/wall_climber/docs/x_plotter_foundation.md)
