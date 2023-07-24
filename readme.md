# DDPG Tutorial

This is the code from [Reinforcement Learning in Continuous Action Spaces | DDPG Tutorial Pytorch](https://www.youtube.com/watch?v=6Yd5WnYls_Y)
by [Machine Learning with Phil](https://www.youtube.com/@MachineLearningwithPhil).

https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/DDPG/pytorch/lunar-lander/ddpg_torch.py

Don't ask me why, but I typed all the code myself, following along with Phil's
video. I made a few adjustments here and there, but it is 99% Phil's code.

Initially, I wanted to make this code into a Jupyter notebook running in Docker,
but that was made difficult by lack of support. OpenAI's gym environment needs a
lot of system-level dependencies installed and I don't like polluting my
development system with dependencies like this. However, Docker does not give
good access to the GPU on my Macbook Air M2, plus rendering gyms is easier
without virtualisation and remoting. Thus, this project comes with a virtual
environment and is intended to be run on bare metal.

## Steps to Implement

Assuming you are on Mac OS X:

```bash
$ brew install swig
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
(venv) $ python torch_lunar_lander.py
```

