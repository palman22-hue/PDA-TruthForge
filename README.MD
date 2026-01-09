# 🤖 PDA TruthForge

[![DOI](https://zenodo.org/badge/1108669370.svg)](https://doi.org/10.5281/zenodo.18010895)

Personal Digital Agent with modular cognition, ethical reasoning, and emotional awareness.

---


## PDA Entropy Simulation

This repository includes a PDA entropy simulation that demonstrates the Coercion–Entropy Principle in practice.

- Location: `Experiments/entropy_sim.py`
- Description: Simulates a 32D PDA state under two scenarios:
  - **Non-coercive (Ethical)**: Applied force aligns with the natural trajectory `N`, keeping total entropy ΔS low, karma positive, and coherence high.
  - **Coercive (Forced)**: Applied force deviates strongly from `N`, causing high entropy, accumulating karma debt, and loss of coherence.
- Output: The script plots *“PDA Entropy Simulation: Coercion vs Non-Coercion”* and saves results as `entropysimrun.png`.

To run the simulation:

```bash
pip install -r requirements.txt  # includes numpy & matplotlib
python Experiments/entropy_sim.py
```

## 🚀 Installation

### 📦 Option 1: Installer (Windows only)
Download the latest release from [GitHub Releases](https://github.com/palman22-hue/PDA-TruthForge/releases)  
Run `PDA_Agent_Installer.exe` and follow the instructions.

> ⚠ Requirements:
> - Python 3.10+
> - [Ollama](https://ollama.com) installed and running
> - Mistral model pulled via `ollama pull mistral`

### 🛠 Option 2: Manual setup
```bash
git clone https://github.com/palman22-hue/PDA-TruthForge
cd PDA
pip install -r requirements.txt
python api.py

📜 Release History
See CHANGELOG.md for full version history.

Latest release: v1.2.0 — PDA TruthForge  
Includes installer, GUI launcher, and universal runtime checks.

🧠 Modules
core/ — cognition: perception, memory, planning, emotions

modules/ — extensions and probes

static/ — web interface

api.py — main entry point

pda_agent_gui.ps1 — GUI launcher

PDA_Agent_Installer.iss — installer script

🛡 License & Credits
This project is open-source and distributed under the GPLv3 License.
Created by Esteban Palman — system architect, storyteller, and agent‑maker.


---



[def]: https://doi.org/10.5281/zenodo.18010895