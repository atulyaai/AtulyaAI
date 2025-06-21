# config.py
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import json
import os

CONFIG_PATH = os.path.abspath('configs/config.json')

class ConfigChangeHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback
    def on_modified(self, event):
        if event.src_path == CONFIG_PATH:
            self.callback()

class ConfigWatcher:
    def __init__(self, config_path=CONFIG_PATH):
        self.config_path = config_path
        self.listeners = []
        self.observer = Observer()
        self.handler = ConfigChangeHandler(self._notify_listeners)
        self.observer.schedule(self.handler, os.path.dirname(config_path), recursive=False)
        self._load_config()
        self._stop_event = threading.Event()

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

    def _notify_listeners(self):
        self._load_config()
        for listener in self.listeners:
            listener(self.config)

    def register_listener(self, listener):
        self.listeners.append(listener)

    def start(self):
        self.observer.start()
        try:
            while not self._stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self._stop_event.set()
        self.observer.stop()
        self.observer.join() 