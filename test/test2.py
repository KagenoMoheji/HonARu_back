import os
import sys
import traceback
import base64
import json
from difflib import SequenceMatcher
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.oauth2 import service_account
from google.protobuf.json_format import MessageToDict
from google.cloud import vision

GCP_CLOUD_VISION_API_URL: str = "https://vision.googleapis.com/v1/images:annotate?key="
GCP_API_KEY: str = <API-KEY>

def img_preprocess(img: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = smoothing_img(img, params=(3, 3))
        return img

def get_paragraphs(res: dict) -> np.ndarray:
    '''
    Vision APIからのレスポンスから検出テキスト(Paragraphs)を抽出する。
    ※WordsはJSONに無さそうだったので取得は厳しそう。

    Args:
        res(dict): Vision APIからのレスポンス。
    Returns:
        paragraphs(np.ndarray): 検出テキスト(Paragraphs)のnumpy配列。
    '''
    pass

def get_boundingBox(res: dict, mode: str) -> np.ndarray:
    '''
    Vision APIからのレスポンスからフレームの頂点座標を抽出する。

    Args:
        res(dict): Vision APIからのレスポンス。
        mode(str): フレームの粒度の選択。p=paragraphs(1単語以上の組み合わせ・フレーズごと)、w=words(単語ごと)。
    Returns:
        frame_list(np.ndarray): 頂点座標集合の2次元numpy配列。
    '''
    pass
    # return frame_list



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
    _, data = cv2.imencode(".jpg", img)
    return base64.b64encode(data)

def img_base642np(img_base64: bytes) -> np.ndarray:
    '''
    base64形式のバイナリデータをnumpy配列に変換する。

    Args:
        img_base64(bytes): 画像のbase64形式のバイナリデータ。
    Returns:
        (np.ndarray): 画像のnumpy配列。
    '''
    img_bytes = base64.b64decode(img_base64)
    img_np = np.fromstring(img_bytes, np.uint8)
    return cv2.imdecode(img_np, cv2.IMREAD_ANYCOLOR) # cv2.IMREAD_UNCHANGED # cv2.IMREAD_COLOR

def smoothing_img(img: np.ndarray, params: tuple = (3, 3), mode: str = "g") -> np.ndarray:
    '''
    平滑化をする。

    Args:
        img(np.ndarray): 画像のnumpy配列。
        params(tupple): 平滑化のパラメータ。modeに依存し必要な長さが変わる。デフォルトは(3, 3)
        mode(str): 平滑化手法の選択。a=移動平均、g=ガウシアン、…。デフォルトは"g"。
    Returns:
        (np.ndarray): 平滑化された画像のnumpy配列。
    '''
    try:
        if mode == "a":
            if len(params) != 2:
                raise IndexError("Too many or less parameters.")
            return cv2.blur(img, params, 0)
        if mode == "g":
            if len(params) != 2:
                raise IndexError("Too many or less parameters.")
            return cv2.GaussianBlur(img, params, 0)
    except Exception as e:
        traceback.print_exc()
        sys.exit()

def binarize_img(img: np.ndarray, threshold: int = 127, threshold_type: int = cv2.THRESH_BINARY) -> np.ndarray:
    '''
    二値化をする。

    Args:
        img(np.ndarray): 画像のnumpy配列。
        threshold(int): 二値化の閾値。デフォルトは127。
        threshold_type(int): cv2.THRESH_<>に基づく平滑化手法の定数。デフォルトはcv2.THRESH_BINARY。
    Returns:
        (np.ndarray): 二値化された画像のnumpy配列。
    '''
    img: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(img, threshold, 255, threshold_type)[1]

def get_polyframed_img(polyframe_list: np.ndarray, filepath: str = "", img: np.ndarray = None, color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    '''
    n(>2)点頂点の座標集合の2次元配列を引数として受け取り、# 簡単にcv2の関数で描写する。
    ただし辺がクロスする場合が発生する。

    Args:
        filepath(str) or img(np.ndarray): フレームを付与する画像データ。
        polyframe_list(np.ndarray): 1つ以上の閉包図形を描写する頂点座標集合の2次元numpy配列。
        color(tuple): BGRカラーコード。デフォルトは緑。
        thickness(int): 線の太さ。デフォルトは2。
    Returns:
        img(np.ndarray): フレームが付与された画像のnumpy配列。
    '''
    try:
        if not filepath and img is None:
            raise ValueError("'get_polyframed_img()' requires filepath OR img.")
        if filepath and img is not None:
            raise ValueError("'get_polyframed_img()' requires filepath OR img.")
        
        if filepath:
            img: np.ndarray = cv2.imread(filepath)
        for pts in polyframe_list:
            img = cv2.polylines(img, [np.int32(pts)], isClosed=True, color=color, thickness=thickness)
        return img
    except Exception as e:
        traceback.print_exc()
        sys.exit()

def get_rectframed_img(rectframe_list: np.ndarray, filepath: str = "", img: np.ndarray = None, color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    '''
    4点頂点のリスト集合を引数として受け取り、get_polyframed_img()よりは丁寧な短形フレームを付与する。

    Args:
        filepath(str) or img(np.ndarray): フレームを付与する画像データ。
        rectframe_list(np.ndarray): 1つ以上の四角形を描写する頂点座標集合の2次元numpy配列。
        color(tuple): BGRカラーコード。デフォルトは緑。
        thickness(int): 線の太さ。デフォルトは2。
    Returns:
        img(np.ndarray): フレームが付与された画像のnumpy配列。
    '''
    try:
        if (not filepath and img is None) or (filepath and img is not None):
            raise ValueError("'get_polyframed_img()' requires filepath OR img.")
        
        if filepath:
            img: np.ndarray = cv2.imread(filepath)
        for pts in rectframe_list:
            tmp_pts: np.ndarray = pts.copy() # 浅いコピーに注意
            if len(tmp_pts) != 4:
                raise IndexError("Too many or less points to make a rectangle.")

            # 四角形になるように四点を並べ替え
            # 重心求めてるけど、めんどいから平均してる
            mid_pt: np.ndarray = np.sum(tmp_pts, axis=0) / 4 # len(tmp_pts)
            i: int = 0
            while i < len(pts):
                j: int = 0
                while j < len(tmp_pts) - 1: # 第3象限はラスト余り物で追加
                    if i == 0 and\
                        tmp_pts[j][0] <= mid_pt[0] and tmp_pts[j][1] <= mid_pt[1]: # 第2象限
                        break
                    if i == 1 and\
                        mid_pt[0] <= tmp_pts[j][0] and tmp_pts[j][1] <= mid_pt[1]: # 第1象限
                        break
                    if i == 2 and\
                        mid_pt[0] <= tmp_pts[j][0] and mid_pt[1] <= tmp_pts[j][1]: # 第4象限
                        break
                    j += 1
                pts[i] = tmp_pts[j]
                tmp_pts = np.delete(tmp_pts, j, axis=0)
                i += 1

            # 直線を4本引く
            j = 1
            for i in range(len(pts)):
                j = i + 1
                if i == 3: # len(pts) - 1
                    j = 0
                img = cv2.line(img, tuple(pts[i]), tuple(pts[j]), color, thickness)
        return img
    except Exception as e:
        traceback.print_exc()
        sys.exit()

def is_similar(usr_text: str, detected_text: str, min_ratio: float) -> bool:
    '''
    検出されたテキストがユーザ入力のテキストとどれだけ類似しているかを算出し、
    指定する最低類似度を超えているか判定する。

    とりま簡易に実装。部分文字列の類似度も一応できる。

    Args:
        usr_text(str): ユーザ入力のテキスト。
        detected_text(str): 検出されたテキスト。
        min_match(int): 最低類似度。
    Returns:
        (boolean): 最低類似度を超えていたらTrue。
    '''
    shoter_text = usr_text if len(usr_text) <= len(detected_text) else detected_text
    longer_text = detected_text if shoter_text == usr_text else usr_text
    shoter_len = len(shoter_text)
    longer_len = len(longer_text)
    
    ratio: float = max([SequenceMatcher(None, shoter_text, longer_text[i:i+shoter_len]).ratio() for i in range(longer_len - shoter_len + 1)])

    if not usr_text or not detected_text: # ゴリゴリ
        ratio = 0.0
    print("{0} <-> {1}: {2}".format(usr_text, detected_text, ratio))
    return ratio >= min_ratio



if __name__ == "__main__":
    os.makedirs("outputs/", exist_ok=True)

    keyword = "プログ"
    filepath: str = <image file name>
    
    img: np.ndarray = load_img_np(filepath)
    # 画像加工
    img = img_preprocess(img)
    cv2.imwrite("outputs/preprocessed.png", img)

    # vision apiの結果を得る
    img_bytes: bytes = cv2.imencode(".jpg", img)[1].tostring()
    # img_base64: bytes = img_np2base64(img) # base64ではダメ
    CREDENT_JSON: str = "<service-account-key>.json" # これはHerokuでは環境変数に登録して読み込む
    creds: service_account.Credentials = service_account.Credentials.from_service_account_file("../.key/{}".format(CREDENT_JSON))
    client: vision.ImageAnnotatorClient = vision.ImageAnnotatorClient(credentials=creds)
    gcp_img: vision.types.Image = vision.types.Image(content=img_bytes) # img_base64
    res: vision.types.AnnotateImageResponse = client.document_text_detection(image=gcp_img) # client.text_detection(image=gcp_img)

    # レスポンスの確認
    with open("outputs/res.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(MessageToDict(res), indent=4))
    texts = res.text_annotations
    for text in texts[1:]:
        print(text.description, type(text.bounding_poly.vertices))

#     texts = 
#     frame_list = 
#     print("""
# ===========================[before]===============================
# {}
# <{}, {}>
# ==================================================================""".format(texts, len(texts), len(frame_list)))
# #     texts, frame_list = extract_by_similarity_to_usr_keyword(texts, frame_list, keyword)
# #     print("""
# # ===========================[after]===============================
# # {}
# # <{}, {}>
# # =================================================================""".format(texts, len(texts), len(frame_list)))

#     img: np.ndarray = get_polyframed_img(frame_list, filepath=filepath)

#     # フレーム付き画像の保存・表示
#     cv2.imwrite("outputs/output.png", img)