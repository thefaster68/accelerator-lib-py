# accelerator-lib-py

A Python toolkit for simplified particle accelerator simulations
(protons/electrons) focusing on dynamics, ideal fields, and Boris-type integration.

## Features (MVP)
- Charged particles with configurable mass/charge
- Ideal E/B kicks, Boris pusher (non-relativistic and relativistic)
- Elements: **Quadrupole**, **Dipole**, RF gap (stub)
- 2D/3D visualizations of trajectories and on-plane density
- Modular structure to extend to realistic devices

## Installation (development)
```bash
git clone git@github.com:USERNAME/accelerator-lib.git
cd accelerator-lib
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
