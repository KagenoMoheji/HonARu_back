'''
modulesフォルダに移動させて実行する
'''

import sys
import json
import requests
import numpy as np
import cv2
from modules.spine_detect import SpineDetect

sd = SpineDetect()
img: np.ndarray = cv2.imread(<image file name>")

API_URL: str = <REST API URL>
headers: dict = {
    "Content-Type": "application/json"
}
data: str = json.dumps({
    "image": sd.img_np2base64(img).decode("utf-8") # 送信する場合のみに限り，文字列キャストまたはutf-8デコードが必要．
})

res: requests.models.Response = requests.post(API_URL, headers=headers, data=data)
result: dict = res.json()
img_base64 = result["image"] # .encode("utf-8") # 無くても良さそう…
img = sd.img_base642np(img_base64)
print(img)
cv2.imwrite("test/res.png", img)