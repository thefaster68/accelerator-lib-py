# accelerator-lib-py

Python toolkit for a simplified simulation of particle accelerators (protons/electrons) with a focus on dynamics, ideal fields and Boris integration

## (MVP)
- Particles with mass/charge configurable
- Ideal E/B push, Boris pusher (non-rel e rel)
- Elements: **Quadrupole**, **Dipole**, gap RF (stub)
- 2D/3D visuals of trajectories and basic colliding sensor
- Modular stucture to extend at real systems

## Installation
```bash
git clone git@github.com:thefaster68/accelerator-lib-py.git
cd accelerator-lib
python -m venv .venv && source
>
>
>
>
> $ cat > README.md << 'EOF'
# accelerator-lib-py

Python toolkit for a simplified simulation of particle accelerators (protons/electrons) with a focus on dynamics, ideal fields and Boris integration

## Caratteristiche (MVP)
- Particles with mass/charge configurable
- Ideal E/B push, Boris pusher (non-rel e rel)
- Elements: **Quadrupole**, **Dipole**, gap RF (stub)
- 2D/3D visuals of trajectories and basic colliding sensor
- Modular stucture to extend at real systems

## Installation
```bash
git clone git@github.com:thefaster68/accelerator-lib-py.git
cd accelerator-lib
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
