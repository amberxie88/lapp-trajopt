# LAPP-TrajOpt Implementation
Official implementation of [Language-Conditioned Path Planning](https://arxiv.org/abs/2308.16893), published in CoRL 2023.

\[[Paper](https://arxiv.org/abs/2308.16893)\] \[[Website](https://amberxie88.github.io/lapp/)\] \[[Video](https://youtu.be/YWJDhd3PXHU)\]

This repository includes an implementation of TrajOpt for integration with LACO. Part of the TrajOpt implementation referenced [this codebase](https://github.com/k-maheshkumar/trajopt_reimpl/).

The repository for training a language-conditioned collision function (LACO) is [here](https://github.com/amberxie88/lapp).

## example command for running w/ PyRep
```sh
vglrun -d :0.0 python3 run.py solver.use_collision=True experiment_folder=EXPERIMENT_FOLDER experiment_name=EXPERIMENT_NAME restore_laco_snapshot_path=LACO_PATH env_name=pyrep_shapenet solver.convexify_max_iteration=5 solver.k=10 device=4 env_cfg=ENV_CFG_PATH problem.min_dist=0.5
```

To swap out your collision prediction network, create a new model in the laco_model directory.

## Citation
```
@inproceedings{
        xie2023languageconditioned,
        title={Language-Conditioned Path Planning},
        author={Amber Xie and Youngwoon Lee and Pieter Abbeel and Stephen James},
        booktitle={7th Annual Conference on Robot Learning},
        year={2023},
        url={https://openreview.net/forum?id=9bK38pUBzU}
}
```