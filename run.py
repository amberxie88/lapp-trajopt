import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import gc
import yaml
from pathlib import Path
from omegaconf import DictConfig, OmegaConf, open_dict

import hydra
import torch
import numpy as np 
import transformers
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True

import utils
import pyrep_env
from envs.flat_world import FlatWorld
from video import TrainVideoRecorder
from logger import Logger

def make_laco_model(cfg):
    if cfg.laco_model.use_mv:
        raise NotImplementedError
    reload_dir = Path(cfg.restore_laco_snapshot_path).parents[1]
    with open(reload_dir / '.hydra/config.yaml', 'r') as f:
        cfg2 = OmegaConf.create(yaml.safe_load(f))
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.reload_dir = str(reload_dir)
        for k, v in cfg2.laco_model.items():
            if k not in cfg.laco_model.keys():
                continue
            if k != "device" and k != "noise":
                cfg.laco_model[k] = v
    return hydra.utils.instantiate(cfg.laco_model)

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        torch.cuda.set_device(cfg.device)
        np.set_printoptions(precision=3)

        # logger
        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)
        
        # misc
        self.timer = utils.Timer()

        # env
        if cfg.env_name == "flat_world":
            self.env = FlatWorld(cfg.env_cfg)
        elif "pyrep" in cfg.env_name:
            self.env = pyrep_env.make(cfg)
            if cfg.solver.use_collision:
                self.laco_model = make_laco_model(cfg)
                model_ckpt = self.load_snapshot(self.cfg.restore_laco_snapshot_path)['model']
                self.laco_model.init_from(model_ckpt)
                print(f"initialized agent from {self.cfg.restore_laco_snapshot_path}")
                model_ckpt.clear()
                gc.collect()
                with torch.no_grad():
                    torch.cuda.empty_cache()
                if "clip16" in cfg.laco_model.obs_encoder_id:
                    self.image_processor = transformers.ViTImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
                elif "clip14" in cfg[model_name].obs_encoder_id:
                    self.image_processor = transformers.ViTImageProcessor.from_pretrained("openai/clip-vit-large-patch14") 
                else:
                    raise NotImplementedError
                self.env.add_laco_model(self.laco_model, self.image_processor)
        else:
            raise NotImplementedError

        self.video_recorder = TrainVideoRecorder(self.work_dir, render_size=self.env.height)

        # init problem 
        if cfg.env_name == "flat_world":
            self.problem = hydra.utils.instantiate(cfg.problem)
        elif "pyrep" in cfg.env_name:
            self.problem, meta = self.env.init_problem()
            self.run_laco_predictions(meta)

        self.solver = hydra.utils.instantiate(cfg.solver)

    def run(self):
        self.run_solution(dict(soln=self.problem.process_solution(self.problem.initial_guess)), vid_name="init")

        self.solver.init(self.env, self.logger, self.problem)
        soln = self.solver.solve()
        soln['soln'] = self.problem.process_solution(soln['soln'])
        self.run_solution(soln)
        #P, q, G, lbG, ubG, initial_guess

    def run_solution(self, soln, vid_name="soln"):
        obs_lst = []
        for idx, state in enumerate(soln['soln']):
            # print(state)
            self.env.set_state(state)
            print(state, [f"{s:.2f}" for s in self.env.env.agent.get_joint_positions()])
            # print(state, self.env.env.agent_ee_tip.get_position())
            if idx == 0:
                self.video_recorder.init(self.env.render())
                # if vid_name == "soln":
                #     for shape in self.env.env.shapes:
                #         shape.set_dynamic(True)
                #         shape.set_respondable(True)
            else:
                self.video_recorder.record(self.env.render())

            from PIL import Image
            out = (self.env.env.render()*255).astype(np.uint8)
            obs_lst.append(out)
            
        self.video_recorder.save(f"{vid_name}.mp4")

        # write solution file to .txt
        with open(self.work_dir / f'{vid_name}.txt', 'w') as f:
            for state in soln['soln']:
                f.write(f"{state}")

        print(self.work_dir)
        if 'collision_info' in soln and soln['collision_info'] is not None:
            print(soln.keys(), soln['collision_info'].keys())
            import imageio
            frames = []
            x = range(len(obs_lst))
            distances = 1-soln['collision_info']['raw_distances']
            gt = soln['collision_info']['gt_collision']
            for idx in range(len(obs_lst)):
                plt.clf()
                fig, ax = plt.subplots(1, 1)
                ax.plot(x[:idx+1], distances[:idx+1], color='red')
                ax.plot(x[:idx+1], gt[:idx+1])
                ax.plot(x[:idx+1], np.ones(idx+1)*(1-self.cfg.problem.min_dist))
                ax.set_xlim([-1, len(x)+1])
                ax.set_ylim([-0.5, 1.5])
                fig.canvas.draw()
                plot_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                plot_data = plot_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                curr_obs = np.pad(obs_lst[idx], ((0, 0), (192, 192), (0, 0)), mode='constant', constant_values=0)
                to_save = np.vstack([curr_obs, plot_data])
                frames.append(to_save)
                plt.close(fig)
            print("len(frames)", len(frames), x)
            imageio.mimsave(f'{self.work_dir}/soln_pred.mp4', frames, fps=1)

    def run_laco_predictions(self, meta):
        collisions, obs, states = meta['all_colls'], meta['all_obs'], meta['all_states']
        collisions_np, obs_np, states = np.array(collisions[1:]), np.array(obs[1:]), np.array(states[1:])
        obs_np = obs_np.transpose(0,3,1,2)
        collisions, obs, states = torch.from_numpy(collisions_np), torch.from_numpy(obs_np), torch.from_numpy(states)
        collisions, obs, states = collisions.to(self.cfg.device), obs.to(self.cfg.device), states.to(self.cfg.device)

        try:
            vit_obs = self.image_processor.preprocess(obs_np[0])['pixel_values'][0]
        except:
            return
        vit_obs = torch.from_numpy(vit_obs).unsqueeze(0).repeat(obs.shape[0], 1, 1, 1).to(self.cfg.device)
        data = dict(vit_obs=vit_obs, collisions=collisions, obs=obs, states=states)
        data['language'] = [self.problem.lang] * obs.shape[0] 
        with torch.no_grad():
            pred = self.laco_model.forward(data)
        x = range(obs.shape[0])
        plt.plot(x, pred.cpu().numpy(), color='red')
        plt.plot(x, collisions_np)
        plt.ylim([-0.5, 1.5])
        plt.savefig(f'{self.work_dir}/demo_pred.png')

        import imageio
        frames = []
        for idx in range(len(x)):
            plt.clf()
            fig, ax = plt.subplots(1, 1)
            ax.plot(x[:idx+1], pred.cpu().numpy()[:idx+1], color='red')
            ax.plot(x[:idx+1], (collisions_np.sum(axis=1)>0).astype(np.int64)[:idx+1])
            ax.plot(x[:idx+1], np.ones(idx+1)*(1-self.cfg.problem.min_dist))
            # print(len(x[:idx]))
            ax.set_xlim([-1, len(x)+1])
            ax.set_ylim([-0.5, 1.5])
            fig.canvas.draw()
            plot_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            plot_data = plot_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            curr_obs = (obs_np[idx].transpose(1,2,0)*255).astype(np.uint8)
            curr_obs = np.pad(curr_obs, ((0, 0), (192, 192), (0, 0)), mode='constant', constant_values=0)
            to_save = np.vstack([curr_obs, plot_data])
            frames.append(to_save)
            plt.close(fig)
        
        imageio.mimsave(f'{self.work_dir}/demo_pred.mp4', frames, fps=20)

    def load_snapshot(self, path):
        snapshot = Path(path)

        with snapshot.open('rb') as f:
            payload = torch.load(f, map_location=lambda storage, loc: storage.cuda(self.cfg.device))
        return payload


@hydra.main(config_path='.', config_name='run')
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()
    # workspace.video_test()


if __name__ == '__main__':
    main()
