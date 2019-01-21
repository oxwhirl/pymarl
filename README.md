```diff
- This is an BETA release
```

# Deep MARL framework

PyMARL is framework for deep multi-agent reinforcement learning and includes implementations of the following algorithms:
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)

PyMARL currently uses [SMAC](https://github.com/oxwhirl/smac) as its environment.

## Installation instructions

Build the Dockerfile using 
```
cd docker
bash build.sh
```

Set up StarCraft II and SMAC.
> bash install_sc2.sh

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Run an experiment 

> python3 src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=2s_3z

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

To run stuff using the Docker container:
> bash run.sh $GPU python3 src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=2s_3z

All results will be stored in the `Results` folder.

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraftII replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraftII. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

The saved replays can be watched by double-clicking on them or using the following command:

> python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay NAME.SC2Replay

(The window size is quite small at the moment, but will be fixed once deepmind accepts my pull request).

## Documentation/Support

Documentation is a little sparse at the moment (but will improve!). Please raise an issue in this repo, or email [Tabish](mailto:tabish.rashid@cs.ox.ac.uk)

## License

Code licensed under the Apache License v2.0
