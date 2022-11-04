import json
import time

import websocket
from websocket import WebSocketConnectionClosedException
import threading

from PoseInterpreter.poseInterpreter import PoseInterpreter

enable_websocket_send = False
allowed_joints = ['head_z']


class PoseInterpreterSimplePyBotSDK(PoseInterpreter):
    MAX_SEND_PER_SECOND = 20

    def __init__(self, config_path: str, host: str, static_image_mode: bool = False,
                 display_face_connections: bool = True, calc_z: bool = False,
                 ws_block_on_error: bool = False):
        super().__init__(config_path, static_image_mode, display_face_connections, calc_z)

        self._websocket_host = host
        self._websocket_simplepybotsdk_app = None
        self._websocket_simplepybotsdk_ws = None
        self._ws_block_on_error = ws_block_on_error
        self._enable_send = False
        self._run_websocket_handler()
        self._last_send = 0

    def _run_websocket_handler(self):
        print("Websocket starting connection with: {}".format(self._websocket_host))
        self._websocket_simplepybotsdk_app = websocket.WebSocketApp(
            self._websocket_host,
            on_open=self._websocket_on_open,
            on_error=self._websocket_on_error,
            on_close=self._websocket_on_close,
            on_message=self._websocket_on_message
        )
        threading.Thread(target=self._websocket_simplepybotsdk_app.run_forever, args=(), daemon=True).start()

    @staticmethod
    def _websocket_on_open(ws):
        print("Websocket established")

    @staticmethod
    def _websocket_on_error(ws, error):
        print("Websocket error: {}".format(error))

    def _websocket_on_close(self, ws):
        print("Websocket closed")
        self._enable_send = False
        self._websocket_simplepybotsdk_app = None

    def _websocket_on_message(self, ws, message):
        print("Websocket message received: {}".format(message))
        ws.send('{"socket": {"format": "block"}}')
        self._websocket_simplepybotsdk_ws = ws
        self._enable_send = True

    def send_ptp_with_websocket(self):
        if not enable_websocket_send:
            print("{} WEBSOCKET SEND DISABLED!!".format(self.computed_ptp))
            return
        if (time.time() - self._last_send) < 1.0 / self.MAX_SEND_PER_SECOND:
            return
        self._last_send = time.time()
        if self._enable_send:
            to_send = {}
            for key, value in self.computed_ptp.items():
                if key in allowed_joints:
                    to_send[key] = value
            print("to_send={} -- computed_ptp={}".format(to_send, self.computed_ptp))
            try:
                payload = {
                    'type': 'C2R',
                    'data': {
                        'area': 'motion',
                        'action': 'ptp',
                        'command': {'seconds': 0.01, **to_send}
                    }
                }
                self._websocket_simplepybotsdk_ws.send(json.dumps(payload))
            except WebSocketConnectionClosedException as e:
                print("Websocket send error: {}".format(e))
        else:
            print("Waiting websocket connection")
            if self._websocket_simplepybotsdk_app is None and self._ws_block_on_error is False:
                self._run_websocket_handler()
            elif self._websocket_simplepybotsdk_app is None and self._ws_block_on_error is True:
                exit(-1)
