import os
import json
import base64
import asyncio
import requests
import cv2
import numpy as np


def load_img_base64(filepath: str) -> bytes:
    '''
    画像ファイルをbase64で読み込む。

    Args:
        filepath(str): ファイルパスまたはファイル名。
    Returns:
        (bytes): base64に変換された画像のバイナリデータ。
    '''
    with open(filepath, "rb") as imf:
        img_byte: bytes = imf.read()
    return base64.b64encode(img_byte)

def load_img_np(filepath: str) -> np.ndarray:
    '''
    画像ファイルをnp.ndarrayで読み込む。

    Args:
        filepath(str): ファイルパスまたはファイル名。
    Returns:
        (np.ndarray): 画像のnumpy配列。
    '''
    return cv2.imread(filepath)

def img_np2base64(img: np.ndarray) -> bytes:
    '''
    numpy配列をbase64形式のバイナリデータに変換する。

    Args:
        img(np.ndarray): 画像のnumpy配列。
    Returns:
        (bytes): base64のバイナリデータ。
    '''
    _, data = cv2.imencode(".jpg", img) # data: bytes
    return base64.b64encode(data)

def img_base642np(img_base64: bytes) -> np.ndarray:
    '''
    base64形式のバイナリデータをnumpy配列に変換する。

    Args:
        img_base64(bytes): 画像のbase64形式のバイナリデータ。
    Returns:
        (np.ndarray): 画像のnumpy配列。
    '''
    img_bytes: bytes = base64.b64decode(img_base64)
    img_np: np.ndarray = np.fromstring(img_bytes, np.uint8)
    return cv2.imdecode(img_np, cv2.IMREAD_ANYCOLOR) # cv2.IMREAD_UNCHANGED # cv2.IMREAD_COLOR



if __name__ == "__main__":
    API_URL: str = <REST API URL>
    headers: dict = {
        "Content-Type": "application/json"
    }

    os.makedirs("outputs/", exist_ok=True)
    keyword = "Java 人工知能"

    resize_ratio = 0.8
    filepath = <video file name>
    cap = cv2.VideoCapture(filepath)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read() # ret: bool, frame: np.ndarray
        if not ret:
            break
        
        frame = cv2.resize(frame, (int(w * resize_ratio), int(h * resize_ratio)))

        img_base64 = img_np2base64(frame)
        data: str = json.dumps({
            "request": {
                "keyword": keyword,
                "image": img_base64.decode("utf-8") # 送信する場合のみに限り，文字列キャストまたはutf-8デコードが必要．
            }
        })



        # このあたりに画像の偶奇で非同期する


        


        result: requests.models.Response = requests.post(API_URL, headers=headers, data=data)
        res: dict = result.json()
        # レスポンスの確認
        with open("outputs/res.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=4)
        if res["image"] == "0":
            print("Null response image")
            cv2.imshow("Frame", frame) # クライアントで取得した画像データで表示
            if cv2.waitKey(1) == 13: break
            continue
        img = img_base642np(res["image"])
        # print(res["itemList"])

        # 表示
        cv2.imshow("Frame", img)
        if cv2.waitKey(1) == 13: break
    cv2.destroyAllWindows()
    cap.release()