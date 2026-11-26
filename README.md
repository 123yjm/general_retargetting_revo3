# General Retargetting — BrainCo Revo 3 (Manus Glove + MuJoCo)

Self-contained real-time human-to-dexterous-hand retargeting visualizer for **BrainCo Revo 3** using a **Manus Glove** (live hardware SDK input), **URMA-Hand (Dexformer)** zero-shot policy, and **MuJoCo**.

## Directory Layout

- `revo3_manus_retarget.py`: Main live retargeting and MuJoCo visualization script
- `run.sh`: Portable launcher script
- `requirements.txt`: Minimal Python dependencies
- `models/stage2_best.pt`: Pretrained URMA-Hand checkpoint (9-hand train + Manus FT, Revo 3 zero-shot)
- `models/v61_all12_stage2_best.pt`: New URMA-Hand checkpoint trained on all 12 hands (including Revo 3)
- `config/revo3_right.yaml`: Revo 3 hand configuration
- `assets/revo3_system/`: Revo 3 URDF and visual meshes
- `retarget_core/`: Self-contained URMA-Hand inference & FK modules
- `vendor/manus_sdk/`: Bundled Manus Python SDK

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Connect Manus USB dongle and power on glove(s), then run:
./run.sh                    # Default: integrated mode (direct dongle), right hand
./run.sh --checkpoint models/v61_all12_stage2_best.pt  # Use the new all-12-hands checkpoint
./run.sh --hand left        # Mirror left Manus glove to right Revo 3
./run.sh --connection local # Connect to an already-running local MANUS Core
```
