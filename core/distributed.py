# distributed.py
import multiprocessing as mp
import socket
import threading
import pickle

class DistributedNode:
    def __init__(self, host='localhost', port=5000):
        self.host = host
        self.port = port
        self.server = None
        self.running = False
        self.handlers = {}

    def start_server(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen(5)
        self.running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self):
        while self.running:
            client, addr = self.server.accept()
            threading.Thread(target=self._handle_client, args=(client,), daemon=True).start()

    def _handle_client(self, client):
        data = b''
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            data += chunk
        try:
            msg = pickle.loads(data)
            cmd = msg.get('cmd')
            if cmd in self.handlers:
                response = self.handlers[cmd](msg)
                client.sendall(pickle.dumps(response))
        except Exception as e:
            client.sendall(pickle.dumps({'error': str(e)}))
        finally:
            client.close()

    def register_handler(self, cmd, handler):
        self.handlers[cmd] = handler

    def send(self, host, port, msg):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            s.sendall(pickle.dumps(msg))
            data = b''
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
            return pickle.loads(data)

    def stop(self):
        self.running = False
        if self.server:
            self.server.close() 