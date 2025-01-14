import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import time
import curses
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

class KeyboardListener(mp.Process):
    def __init__(self, 
            shm_manager, 
            get_max_k=30, 
            frequency=10,
            max_value=5, 
            deadzone=(0,0,0,0,0,0), 
            dtype=np.float32,
            n_buttons=2,
            ):
        """
        Continuously listen to keyboard events
        and update the latest state.

        max_value: {300, 500} 300 for wired version and 500 for wireless
        deadzone: [0,1], number or tuple, axis with value lower than this value will stay at 0
        
        front
        z
        ^   _
        |  (O) keyboard
        |
        *----->x right
        y
        """
        super().__init__()
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()

        # copied variables
        self.frequency = frequency
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone
        self.n_buttons = n_buttons
        self.tx_zup_spnav = np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ], dtype=dtype)

        example = {
            # 3 translation, 3 rotation, 1 period
            'motion_event': np.zeros((7,), dtype=np.int64),
            # left and right button
            'button_state': np.zeros((n_buttons,), dtype=bool),
            'receive_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager, 
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        # shared variables
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.ring_buffer = ring_buffer

    # ======= get state APIs ==========

    def get_motion_state(self):
        state = self.ring_buffer.get()
        state = np.array(state['motion_event'][:6], 
            dtype=self.dtype) / self.max_value
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state
    
    def get_motion_state_transformed(self):
        """
        Return in right-handed coordinate
                    z
        x<----------* 
                    |   
         Test board |  
                    v
                    y
                robot

        """
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_zup_spnav @ state[:3]
        tf_state[3:] = self.tx_zup_spnav @ state[3:]
        return tf_state

    def get_button_state(self):
        state = self.ring_buffer.get()
        return state['button_state']
    
    def is_button_pressed(self, button_id):
        return self.get_button_state()[button_id]
    
    #========== start stop API ===========

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait()
    
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= main loop ==========
    def run(self):
        def curses_main(stdscr):
            curses.cbreak()
            stdscr.nodelay(True)
            stdscr.keypad(True)
            stdscr.clear()

            motion_event = np.zeros((7,), dtype=np.int64)
            button_state = np.zeros((self.n_buttons,), dtype=bool)
            # send one message immediately so client can start reading
            self.ring_buffer.put({
                'motion_event': motion_event,
                'button_state': button_state,
                'receive_timestamp': time.time()
            })
            self.ready_event.set()

            try:
                while not self.stop_event.is_set():
                    key = stdscr.getch()
                    if key == curses.ERR:
                        time.sleep(1/self.frequency)
                        continue

                    if key == ord('w'):
                        motion_event[2] = self.max_value
                    elif key == ord('s'):
                        motion_event[2] = -self.max_value
                    elif key == ord('a'):
                        motion_event[0] = -self.max_value
                    elif key == ord('d'):
                        motion_event[0] = self.max_value
                    elif key == ord('q'):
                        motion_event[1] = self.max_value
                    elif key == ord('e'):
                        motion_event[1] = -self.max_value
                    elif key == ord('w') or key == ord('s'):
                        motion_event[2] = 0
                    elif key == ord('a') or key == ord('d'):
                        motion_event[0] = 0
                    elif key == ord('q') or key == ord('e'):
                        motion_event[1] = 0

                    receive_timestamp = time.time()
                    self.ring_buffer.put({
                        'motion_event': motion_event,
                        'button_state': button_state,
                        'receive_timestamp': receive_timestamp
                    })
                    time.sleep(1/self.frequency)
            finally:
                curses.nocbreak()
                stdscr.keypad(False)
                stdscr.nodelay(False)
                curses.endwin()

        curses.wrapper(curses_main)

if __name__ == "__main__":
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    keyboard_listener = KeyboardListener(shm_manager=shm_manager)
    keyboard_listener.start()
    shm_manager.shutdown()
    # try:
    #     while True:
    #         motion_state = keyboard_listener.get_motion_state()
    #         button_state = keyboard_listener.get_button_state()
    #         print(f"Motion State: {motion_state}, Button State: {button_state}")
    #         time.sleep(0.1)
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     keyboard_listener.stop()