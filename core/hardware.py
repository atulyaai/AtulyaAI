# hardware.py
import threading
import time
import os

class Device:
    def __init__(self, name, device_type, control_fn, stream_fn=None):
        self.name = name
        self.device_type = device_type
        self.control_fn = control_fn
        self.stream_fn = stream_fn
    def send_command(self, command):
        try:
            return self.control_fn(command)
        except Exception as e:
            return f'Error: {e}'
    def stream(self):
        if self.stream_fn:
            try:
                return self.stream_fn()
            except Exception as e:
                return f'Error: {e}'
        return None

class HardwareManager:
    def __init__(self):
        self.devices = {}
    def register_device(self, device):
        self.devices[device.name] = device
    def control_device(self, name, command):
        if name in self.devices:
            return self.devices[name].send_command(command)
        return f'Device {name} not found.'
    def list_devices(self):
        return list(self.devices.keys())
    def poll_devices(self, interval=1):
        def poll():
            while True:
                for device in self.devices.values():
                    device.send_command('status')
                time.sleep(interval)
        threading.Thread(target=poll, daemon=True).start()
    def discover_devices(self, search_path='/dev'):
        # Plug-and-play device discovery (example for Linux)
        found = []
        if os.path.exists(search_path):
            for f in os.listdir(search_path):
                if 'tty' in f or 'usb' in f:
                    found.append(f)
        return found
    def stream_all(self):
        streams = {}
        for name, device in self.devices.items():
            streams[name] = device.stream()
        return streams
    def federated_update(self, model_update):
        # Placeholder: integrate edge/federated learning logic
        return 'Federated update received.' 