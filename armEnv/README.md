# OpenAI Gym Environment for Intel Robot Arm

Contains the ArmDOF_0 Model used in Jason and Michal's 2017 UROP report

### Prerequisites

[OpenAI Gym](https://github.com/openai/gym) + 
[MuJoCo](http://www.mujoco.org/) + 
[mujoco-py](https://github.com/openai/mujoco-py) + 
[TRPO](https://github.com/joschu/modular_rl)

### Installation
```bash
cd armEnv
pip install -e .
```
May need to use sudo to install


### Verify Installation
Try to import it in python:
```python
import armDOF_0
```

### Run TRPO Training
Go to the TRPO directory, add the following line in `run_pg.py`
```python
import armDOF_0
```
Start training:
```bash
python run_pg.py --gamma=0.995 --lam=0.98 --agent=modular_rl.agentzoo.TrpoAgent
--max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=500 --seed=0 --hid_sizes=16,16
--timesteps_per_batch=3000 --env=ArmDOF_0-v0 --outfile=./result
```
The training videos will be saved in `result.dir` directory



