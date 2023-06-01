import datetime
import random


class IdentifierFactory:

    def __init__(self):
        self._time = datetime.datetime.now(tz=datetime.timezone.utc)
        self._formatted_time = self._time.isoformat()
        self._uuid = random.getrandbits(64) | (1 << 63)
        self._run_id = (self._uuid << 64) | int(self._time.timestamp())
        self._encoded_id = hex(self._run_id)[2:]

    def generate_id(self, description: id):
        return f'|iso-time={self._formatted_time}|run-id={self._encoded_id}|description={description}|'
