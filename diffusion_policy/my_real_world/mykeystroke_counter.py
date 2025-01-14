import curses
from collections import defaultdict
from threading import Lock, Thread

class KeystrokeCounter:
    def __init__(self):
        self.key_count_map = defaultdict(lambda: 0)
        self.key_press_list = list()
        self.lock = Lock()
        self.running = True
        stdscr = curses.initscr()
        stdscr.clear()
        self.stdscr = stdscr
        self.stdscr.nodelay(True)
        self.stdscr.timeout(100)
        self.thread = Thread(target=self.capture_keys)
        self.thread.start()

    def capture_keys(self):
        while self.running:
            key = self.stdscr.getch()
            if key != curses.ERR:
                with self.lock:
                    self.key_count_map[key] += 1
                    self.key_press_list.append(key)

    def clear(self):
        with self.lock:
            self.key_count_map = defaultdict(lambda: 0)
            self.key_press_list = list()

    def __getitem__(self, key):
        with self.lock:
            return self.key_count_map[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def get_press_events(self):
        with self.lock:
            events = [chr(key) for key in self.key_press_list]
            self.key_press_list = list()
            return events

    def stop(self):
        self.running = False
        self.thread.join()

    def get_key_state(self, key_name):
        key_code = ord(key_name) if len(key_name) == 1 else getattr(curses, f'KEY_{key_name.upper()}', None)
        if key_code is None:
            raise ValueError(f"Unknown key name: {key_name}")
        with self.lock:
            return self.key_count_map[key_code]

def main(stdscr):
    counter = KeystrokeCounter()
    try:
        while True:
            stdscr.clear()
            stdscr.addstr(0, 0, f'Space: {counter.get_key_state(" ")}')
            stdscr.addstr(1, 0, f'q: {counter.get_key_state("q")}\n')
            stdscr.refresh()
            curses.napms(1000 // 60)
    except KeyboardInterrupt:
        counter.stop()
        events = counter.get_press_events()
        stdscr.clear()
        stdscr.addstr(0, 0, str(events))
        stdscr.refresh()
        stdscr.getch()

if __name__ == '__main__':
    curses.wrapper(main)
