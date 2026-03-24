from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, Logger

import numpy as np
import torch


def _set_command_range(env_cfg, name, value):
    if hasattr(env_cfg.commands.ranges, name):
        setattr(env_cfg.commands.ranges, name, [value, value])


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Override task settings to create a deterministic stand-still scene.
    env_cfg.env.num_envs = 1 if args.num_envs is None else min(env_cfg.env.num_envs, args.num_envs)
    env_cfg.env.reference_state_initialization = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.measure_heights = False
    env_cfg.noise.add_noise = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    _set_command_range(env_cfg, "lin_vel_x", 0.0)
    _set_command_range(env_cfg, "lin_vel_y", 0.0)
    _set_command_range(env_cfg, "ang_vel_yaw", 0.0)
    _set_command_range(env_cfg, "heading", 0.0)
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False

    if hasattr(train_cfg.runner, "amp_num_preload_transitions"):
        train_cfg.runner.amp_num_preload_transitions = 1

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs = env.get_observations()
    env.commands[:] = 0.0

    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    stop_state_log = 100
    stop_rew_log = env.max_episode_length + 1
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)

    for i in range(10 * int(env.max_episode_length)):
        obs, _, rews, dones, infos, _, _ = env.step(actions)
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(
                    LEGGED_GYM_ROOT_DIR,
                    'logs',
                    train_cfg.runner.experiment_name,
                    'exported',
                    'frames',
                    f"{img_idx}.png",
                )
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            dof_target = env.default_dof_pos[robot_index, joint_index].item()
            logger.log_states(
                {
                    'dof_pos_target': dof_target,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == '__main__':
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)