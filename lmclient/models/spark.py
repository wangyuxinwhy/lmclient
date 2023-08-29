from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from datetime import datetime
from time import mktime
from typing import Any
from urllib.parse import urlencode, urlparse
from wsgiref.handlers import format_date_time

import websocket

from lmclient.models.base import BaseChatModel


class SparkChat(BaseChatModel):
    def __init__(
        self,
        app_id: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        spark_url: str | None = None,
    ) -> None:
        self.app_id = app_id or os.environ['SPARK_APP_ID']
        self.api_key = api_key or os.environ['SPARK_API_KEY']
        self.api_secret = api_secret or os.environ['SPARK_API_SECRET']
        self.spark_url = spark_url or os.environ['SPARK_URL']

        self.response: dict[str, Any] = {}
        self.receive_round = 0

    @property
    def host(self) -> str:
        return urlparse(self.spark_url).netloc

    @property
    def path(self) -> str:
        return urlparse(self.spark_url).path

    def generate_new_request_url(self):
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = 'host: ' + self.host + '\n'
        signature_origin += 'date: ' + date + '\n'
        signature_origin += 'GET ' + self.path + ' HTTP/1.1'

        signature_sha = hmac.new(
            self.api_secret.encode('utf-8'), signature_origin.encode('utf-8'), digestmod=hashlib.sha256
        ).digest()
        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        values = {'authorization': authorization, 'date': date, 'host': self.host}
        # 拼接鉴权参数，生成url
        url = self.spark_url + '?' + urlencode(values)
        return url

    def on_message(self, ws_app: websocket.WebSocketApp, message: str) -> None:
        receive_reponse = json.loads(message)

        code = receive_reponse['header']['code']
        if code != 0:
            ws_app.close()
            raise Exception(f'Error code: {code}')

        status = receive_reponse['payload']['choices']['status']
        if status == 2:
            ws_app.close()
            return

        self.response[f'round_{self.receive_round}'] = receive_reponse
        self.receive_round += 1
