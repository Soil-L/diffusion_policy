import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from .tcpsock import TCP_SOCKET
import sys

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

class RobotInterface(TCP_SOCKET):
    """Direct interface to control the real robot."""
    def __init__(self):
        TCP_SOCKET.__init__(self)
        self.robot_arm_server_ip = '192.168.0.10'
        self.robot_arm_server_port = 10003

    def apply_actions(self, actions: dict):
        assert isinstance(actions, dict), "Actions must be a dict"
        message = ','.join(actions.values()) + ',;'
        success, response = self.send_tcp_request(self.robot_arm_server_ip, self.robot_arm_server_port, message)
        if not success:
            # print("[ERROR]: Failed to send TCP request")
            sys.exit(1)
        if b'ok' not in response.lower():
            # print("[ERROR]: Server response does not contain 'ok'")
            # print(f"Server response was: {response}")
            # sys.exit(1)
            pass

        # Wait until the robot has stopped moving
        while True:
            state_message = 'ReadRobotState,0,;'
            success, state_response = self.send_tcp_request(self.robot_arm_server_ip, self.robot_arm_server_port, state_message)
            if not success:
                print("[ERROR]: Failed to read robot state")
                sys.exit(1)
            if b'fail' in state_response.lower():
                print("[ERROR]: Failed to read robot state")
                sys.exit(1)
            
            state_values = state_response.decode().split(',')
            nMovingState = int(state_values[2])
            if nMovingState == 0:
                break
            time.sleep(0.1)  # Wait a bit before checking again
        return
    
    def apply_waypoint(self, actions: dict):
        assert isinstance(actions, dict), "Actions must be a dict"
        message = 'WayPoint,0,{dX},{dY},{dZ},{dRx},{dRy},{dRz},0,0,0,0,0,0,TCP,Base,{dVelocity},{dAcc},0,1,0,0,0,0,0,;'.format(
            dX=actions['dX'],
            dY=actions['dY'],
            dZ=actions['dZ'],
            dRx=actions['dRx'],
            dRy=actions['dRy'],
            dRz=actions['dRz'],
            dVelocity=actions['dVelocity'],
            dAcc=actions['dAcc']
        )
        success, response = self.send_tcp_request(self.robot_arm_server_ip, self.robot_arm_server_port, message)
        if not success:
            sys.exit(1)
        if b'OK' not in response.upper():
            pass

        while True:
            state_message = 'ReadRobotState,0,;'
            success, state_response = self.send_tcp_request(self.robot_arm_server_ip, self.robot_arm_server_port, state_message)
            if not success:
                print("[ERROR]: Failed to read robot state")
                sys.exit(1)
            if b'FAIL' in state_response.upper():
                print("[ERROR]: Failed to read robot state")
                sys.exit(1)
            
            state_values = state_response.decode().split(',')
            nMovingState = int(state_values[2])
            if nMovingState == 0:
                break
            time.sleep(0.1)
        # print("Waypoint applied")
        return

    def read_current_tool_pose(self):
        message = 'ReadCurTCP,0,;'
        success, response = self.send_tcp_request(self.robot_arm_server_ip, self.robot_arm_server_port, message)
        if not success:
            print("[ERROR]: Failed to send TCP request")
            sys.exit(1)
        if b'fail' in response.lower():
            print("[ERROR]: Failed to read current tool pose")
            error_code = response.decode().split(',')[2]
            print(f"Error code: {error_code}")
            sys.exit(1)
        
        pose_values = response.decode().split(',')[2:8]
        pose = [float(value) for value in pose_values]
        return pose

    def read_current_user_pose(self):
        message = 'ReadCurUCS,0,;'
        success, response = self.send_tcp_request(self.robot_arm_server_ip, self.robot_arm_server_port, message)
        if not success:
            print("[ERROR]: Failed to send TCP request")
            sys.exit(1)
        if b'fail' in response.lower():
            print("[ERROR]: Failed to read current user pose")
            error_code = response.decode().split(',')[2]
            print(f"Error code: {error_code}")
            sys.exit(1)
        
        pose_values = response.decode().split(',')[2:8]
        pose = [float(value) for value in pose_values]
        return pose
    
    def read_pose_in_base_frame(self):
        message = 'ReadTCPByName,0,Base,;'
        success, response = self.send_tcp_request(self.robot_arm_server_ip, self.robot_arm_server_port, message)
        if not success:
            print("[ERROR]: Failed to send TCP request")
            sys.exit(1)
        if b'fail' in response.lower():
            print("[ERROR]: Failed to read TCP by name")
            error_code = response.decode().split(',')[2]
            print(f"Error code: {error_code}")
            sys.exit(1)

        pose_values = response.decode().split(',')[2:8]
        pose = [float(value) for value in pose_values]
        return pose
    
    def read_ucs_by_name(self):
        message = f'ReadUCSByName,0,Base,;'
        success, response = self.send_tcp_request(self.robot_arm_server_ip, self.robot_arm_server_port, message)
        if not success:
            print("[ERROR]: Failed to send TCP request")
            sys.exit(1)
        if b'fail' in response.lower():
            print("[ERROR]: Failed to read UCS by name")
            error_code = response.decode().split(',')[2]
            print(f"Error code: {error_code}")
            sys.exit(1)

        pose_values = response.decode().split(',')[2:8]
        pose = [float(value) for value in pose_values]
        return pose
    
    def read_space_pose(self):
        message = 'ReadActPos,0,;'
        success, response = self.send_tcp_request(self.robot_arm_server_ip, self.robot_arm_server_port, message)
        if not success:
            # print("[ERROR]: Failed to send TCP request")
            sys.exit(1)
        if b'fail' in response.lower():
            # print("[ERROR]: Failed to read space pose")
            error_code = response.decode().split(',')[2]
            # print(f"Error code: {error_code}")
            sys.exit(1)

        pose_values = response.decode().split(',')[8:14]
        pose = [float(value) for value in pose_values]
        return pose

    def close(self):
        self.close_socket()

class myInterpolationController(mp.Process):
    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 robot_ip,
                 frequency=125,
                 lookahead_time=0.1,
                 gain=300,
                 max_pos_speed=5, # mm/s
                 max_rot_speed=0.1, # rad/s
                 launch_timeout=3,
                 tcp_offset_pose=None,
                 payload_mass=None,
                 payload_cog=None,
                 joints_init=None,
                 joints_init_speed=1.05,
                 soft_real_time=False,
                 verbose=False,
                 receive_keys=None,
                 get_max_k=128):
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:
            assert 0 <= payload_mass <= 5
        if payload_cog is not None:
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        super().__init__(name="myPositionalController")
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose

        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        if receive_keys is None:
            receive_keys = [
                'ActualTCPPose',
                'ActualTCPSpeed',
                'ActualQ',
                'ActualQd',
                'TargetTCPPose',
                'TargetTCPSpeed',
                'TargetQ',
                'TargetQd'
            ]
        example = dict()
        for key in receive_keys:
            example[key] = np.zeros((6,))
        example['robot_receive_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[myPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def servoL(self, pose, duration=0.1):
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)

    def schedule_waypoint(self, pose, target_time):
        assert target_time > time.time()
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def run(self):
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        robot = RobotInterface()

        try:
            if self.verbose:
                print(f"[myPositionalController] Connect to robot: {self.robot_ip}")

            dt = 1. / self.frequency
            curr_pose = robot.read_space_pose()
            curr_pose[3:6] = st.Rotation.from_euler('xyz', curr_pose[3:6], degrees=True).as_rotvec()
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )

            iter_idx = 0
            keep_running = True
            while keep_running:
                t_start = time.perf_counter()

                t_now = time.monotonic()
                pose_command = pose_interp(t_now)
                # Convert rotation vector to RX, RY, RZ
                rotation = st.Rotation.from_rotvec(pose_command[3:6])
                rX, rY, rZ = rotation.as_euler('xyz', degrees=True)
                pose_command[3] = rX
                pose_command[4] = rY
                pose_command[5] = rZ
                # actions = {
                #     'command': 'MoveRelL',
                #     'nRbtID': '0',
                #     'nAxisId': '2',
                #     'nDirection': '1',
                #     'dDistance': str(pose_command[2]-curr_pose[2]),
                #     'nToolMotion': '0',
                # }
                # robot.apply_actions(actions)
                actions = {
                    'dX': pose_command[0],
                    'dY': pose_command[1],
                    'dZ': pose_command[2],
                    'dRx': pose_command[3],
                    'dRy': pose_command[4],
                    'dRz': pose_command[5],
                    'dVelocity': self.max_pos_speed,
                    'dAcc': 100,
                }
                # print(f"iter_idx:{iter_idx}:  " + str(pose_command))
                robot.apply_waypoint(actions)

                state = dict()
                for key in self.receive_keys:
                    state[key] = np.zeros((6,))
                state['robot_receive_timestamp'] = time.time()
                self.ring_buffer.put(state)

                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    elif cmd == Command.SERVOL.value:
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[myPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_pose[3:6] = st.Rotation.from_euler('xyz', target_pose[3:6], degrees=True).as_rotvec()
                        target_time = float(command['target_time'])
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                time.sleep(max(0, dt - (time.perf_counter() - t_start)))

                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[myPositionalController] Actual frequency {1/(time.perf_counter() - t_start)}")

        finally:
            robot.close()
            self.ready_event.set()

            if self.verbose:
                print(f"[myPositionalController] Disconnected from robot: {self.robot_ip}")
    def read_current_tool_pose(self):
        robot = RobotInterface()
        try:
            pose = robot.read_current_tool_pose()
        finally:
            robot.close()
        return pose
    
    def read_current_user_pose(self):
        robot = RobotInterface()
        try:
            pose = robot.read_current_user_pose()
        finally:
            robot.close()
        return pose
    
    def read_pose_in_base_frame(self):
        robot = RobotInterface()
        try:
            pose = robot.read_pose_in_base_frame()
        finally:
            robot.close()
        return pose
    
    def read_ucs_by_name(self):
        robot = RobotInterface()
        try:
            pose = robot.read_ucs_by_name()
        finally:
            robot.close()
        return pose
    
    def read_space_pose(self):
        robot = RobotInterface()
        try:
            pose = robot.read_space_pose()
        finally:
            robot.close()
        return pose