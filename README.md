# MA_TRANSSHIP: Multi-Agent Reinforcement Learning for Inventory Management

MA_TRANSSHIP is a multi-agent reinforcement learning framework for inventory management with transshipment. It focuses on solving inventory optimization problems with multiple lead times and lateral transshipment between agents.

## Features

- Multi-agent inventory management with transshipment support
- Multiple lead times handling
- Proactive and reactive transshipment modes
- Various demand generation methods (kim_merton, shanshu)
- Multiple product allocation mechanisms (baseline, anupindi)
- Based on HAPPO (Heterogeneous-Agent Proximal Policy Optimization) algorithm

## Requirements

See requirements.txt for detailed dependencies.

## Quick Start

### Training 
Train with default config

```bash
python train.py --load_config tuned_configs/multi_lt_transship/fine_tuned_final.json
```

Train with custom config
```bash
python train.py --load_config path/to/your/config.json --load_change_config path/to/your/change_config.json
```


## Project Structure
harl/
├── envs/
│ └── multi_lt_transship/
│ ├── dist_mechanism/ # Transshipment mechanisms
│ ├── generator.py # Demand generators
│ ├── baseline.py # Baseline policies
│ └── transship_multi_lt.py # Main environment
├── train.py # Training script


## Configuration

The system can be configured through JSON configuration files. Key configuration options include:
- Demand generation parameters
- Transshipment mechanisms
- Agent network architectures
- Training hyperparameters

## License

This project is licensed under the MIT License - see the LICENSE file for details.
