import json
import os.path
import hashlib

class CheckpointManager:

    def __init__(self, filename: str):
        with open(filename) as file:
            payload = file.read()
        ident = hashlib.sha256(payload.encode()).hexdigest()
        self._file = f'checkpoint_{ident}.json'
        self._script = json.loads(payload)
        if not os.path.exists(self._file):
            with open(self._file, 'w') as file:
                json.dump({'checkpoint': 0}, file)
        with open(self._file) as file:
            self._state = json.load(file)['checkpoint']

    def invalidate(self):
        self._state = 0
        with open(self._file, 'w') as file:
            json.dump({'checkpoint': self._state}, file)

    def get_auth(self):
        return self._script['auth']

    def commands(self):
        for item in self._script['script'][self._state:]:
            yield item
            self._state += 1
            with open(self._file, 'w') as file:
                json.dump({'checkpoint': self._state}, file)
