# DDPG Tutorial

This is the code from [Reinforcement Learning in Continuous Action Spaces | DDPG Tutorial Pytorch](https://www.youtube.com/watch?v=6Yd5WnYls_Y)
by [Machine Learning with Phil](https://www.youtube.com/@MachineLearningwithPhil).

Don't ask me why, but I typed all the code myself, following along with Phil's
video. I made a few adjustments here and there, but it is 99% Phil's code. For
comparison, here is [Phil's actual code](https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/DDPG/pytorch/lunar-lander/ddpg_torch.py), which I did use to fix some typos I made.

Special thanks to Phil for personally helping me fix some bugs in my code.

## Getting Started

Initially, I wanted to make this code into a Jupyter notebook running in Docker,
but that was made difficult by lack of support. OpenAI's gym environment needs a
lot of system-level dependencies installed and I don't like polluting my
development system with dependencies like this. However, Docker does not give
good access to the GPU on my Macbook Air M2, plus rendering gyms is easier
without virtualisation and remoting. Thus, this project comes with a virtual
environment and is intended to be run on bare metal.

### Setting up the Environment

Assuming you are on Mac OS X:

```bash
$ brew install swig
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
```

### Training the Model

This model runs at about 10 seconds per episode, after 1000 generations, on my
Macbook Air M2. Checkpoints are saved in a directory named `checkpoints` and the
model training will try to load previous checkpoints before starting to train.

```bash
(venv) $ python torch_lunar_lander.py
```

### Seeing the Trained Model

Training is headless, so after (or even during) training, you may want to see
the model controlling the lunar lander. To do so, start the command below. That
loads the models from the `checkpoints` and runs the game with visuals enabled.

```bash
(venv) $ python visual_lunar_lander.py
```

