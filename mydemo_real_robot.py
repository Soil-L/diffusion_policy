"""
Usage:
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_ur5>

Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from diffusion_policy.my_real_world.myreal_env import RealEnv
from diffusion_policy.my_real_world.keyboard import KeyboardListener
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.my_real_world.mykeystroke_counter import KeystrokeCounter


def start_recording(env, key_counter, t_start, iter_idx, dt):
    env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
    key_counter.clear()
    print('Recording!')

def stop_recording(env, key_counter):
    env.end_episode()
    key_counter.clear()
    print('Stopped.')

def move_robot(env, key_stroke, t_command_target, stage=None):
    step = 10  # Define a uniform step size
    move_map = {
        'r': [0, 0, step, 0, 0, 0],
        'f': [0, 0, -step, 0, 0, 0],
        'a': [step, 0, 0, 0, 0, 0],
        'd': [-step, 0, 0, 0, 0, 0],
        's': [0, step, 0, 0, 0, 0],
        'w': [0, -step, 0, 0, 0, 0]
    }
    target_pose = np.array(move_map[key_stroke])
    env.exec_actions(
        actions=[target_pose], 
        timestamps=[t_command_target - time.monotonic() + time.time()],
        stages=[stage]
    )
    print(f'Press "{key_stroke}" to move.')

def delete_episode(env, key_counter):
    print('Press "y" to confirm deletion of the most recent episode.')
    confirm_key = None
    while confirm_key not in ['y', 'n']:
        confirm_key = key_counter.get_press_events()
        if 'y' in confirm_key:
            env.drop_episode()
            key_counter.clear()
            print('Episode deleted.')
        elif 'n' in confirm_key:
            print('Deletion cancelled.')
            key_counter.clear()



@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(output, robot_ip, vis_camera_idx, init_joints, frequency, command_latency):
    dt = 1/frequency
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            RealEnv(
                output_dir=output, 
                robot_ip=robot_ip, 
                # recording resolution
                obs_image_resolution=(1280,720),
                frequency=frequency,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager
            ) as env:
            cv2.setNumThreads(1)

            # realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)

            time.sleep(1.0)
            print('Ready!')
            state = env.get_robot_state()
            target_pose = state['TargetTCPPose']
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            stage = ''
            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # pump obs
                obs = env.get_obs()

                # handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == 'p':
                        stop = True
                    elif key_stroke == 'j':
                        start_recording(env, key_counter, t_start, iter_idx, dt)
                        is_recording = True
                    elif key_stroke == 'k':
                        stop_recording(env, key_counter)
                        is_recording = False
                    elif key_stroke in ['r', 'f', 'a', 'd', 'w', 's']:
                        move_robot(env, key_stroke, t_command_target, stage)
                    elif key_stroke == 'l':
                        delete_episode(env, key_counter)
                stage = key_counter.get_key_state(' ')

                # visualize
                vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()
                episode_id = env.replay_buffer.n_episodes
                text = f'Episode: {episode_id}, Stage: {stage}'
                if is_recording:
                    text += ', Recording!'
                cv2.putText(
                    vis_img,
                    text,
                    (10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255,255,255)
                )

                cv2.imshow('default', vis_img)
                cv2.pollKey()

                precise_wait(t_sample)
                
                precise_wait(t_cycle_end)

                iter_idx += 1

# %%
if __name__ == '__main__':
    main()
