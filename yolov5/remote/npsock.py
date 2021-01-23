"""
Adapted From https://github.com/sabjorn/NumpySocket
"""
from typing import Dict

import numpy as np
from io import BytesIO


class NpSocketBase:

    @classmethod
    def wrap_arrays(cls, arr: Dict[str, np.array]) -> bytearray:
        f = BytesIO()
        np.savez(f, **arr)
        packet_size = len(f.getvalue())
        header = '{0}:'.format(packet_size)
        out = bytearray(header, 'utf-8')

        f.seek(0)
        out += f.read()

        return out

    @classmethod
    def unwrap_arrays(cls, request):
        frameBuffer = bytearray()
        length = None
        while True:
            data = request.recv(1024)
            frameBuffer += data
            if len(frameBuffer) == length:
                break
            while True:
                if length is None:
                    if b':' not in frameBuffer:
                        break
                    # remove the length bytes from the front of frameBuffer
                    # leave any remaining bytes in the frameBuffer!
                    length_str, ignored, frameBuffer = frameBuffer.partition(b':')
                    length = int(length_str)
                if len(frameBuffer) < length:
                    break
                # split off the full message from the remaining bytes
                # leave any remaining bytes in the frameBuffer!
                frameBuffer = frameBuffer[length:]
                length = None
                break
        # noinspection PyTypeChecker
        return np.load(BytesIO(frameBuffer))
