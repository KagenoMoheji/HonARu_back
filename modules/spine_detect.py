import os
import sys
import traceback
import base64
import json
from difflib import SequenceMatcher
import requests
import cv2
import numpy as np

class SpineDetect:
    GCP_CLOUD_VISION_API_URL: str = "https://vision.googleapis.com/v1/images:annotate?key="
    GCP_API_KEY: str = ""
    def __init__(self):
        self.GCP_API_KEY = os.environ["GCP_API_KEY"]

    def img_preprocess(self, img: np.ndarray, scale_rate: float = 1.0) -> np.ndarray:
        if scale_rate != 1.0:
            h, w = img.shape[:2] # h: int, w: int
            img = cv2.resize(img, (int(w * scale_rate), int(h * scale_rate)))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = self.smoothing_img(img, params=(3, 3))
        return img

    def load_img_base64(self, filepath: str) -> bytes:
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

    def load_img_np(self, filepath: str) -> np.ndarray:
        '''
        画像ファイルをnp.ndarrayで読み込む。

        Args:
            filepath(str): ファイルパスまたはファイル名。
        Returns:
            (np.ndarray): 画像のnumpy配列。
        '''
        return cv2.imread(filepath)

    def img_np2base64(self, img: np.ndarray) -> bytes:
        '''
        numpy配列をbase64形式のバイナリデータに変換する。

        Args:
            img(np.ndarray): 画像のnumpy配列。
        Returns:
            (bytes): base64のバイナリデータ。
        '''
        _, data = cv2.imencode(".jpg", img) # data: bytes?
        return base64.b64encode(data)

    def img_base642np(self, img_base64: bytes) -> np.ndarray:
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

    def binarize_img(self, img: np.ndarray, threshold: int = 127, threshold_type: int = cv2.THRESH_BINARY) -> np.ndarray:
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

    def smoothing_img(self, img: np.ndarray, params: tuple = (3, 3), mode: str = "g") -> np.ndarray:
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

    def request_cloud_vision_api(self, img_base64: bytes, mode: str = "text") -> dict:
        '''
        Cloud Vision APIにリクエストを送って文字検出の結果を得る。

        Args:
            img_base64(bytes): 画像のbase64に基づくバイナリデータ。
            mode(str): GCP Vision APIにおけるテキスト検出の"type"の値。
        Returns:
            (dict): dict型にしたレスポンスのJSON。
        '''
        try:
            detect_type: str = ""
            if mode == "text":
                detect_type = "TEXT_DETECTION"
            elif mode == "doc":
                detect_type = "DOCUMENT_TEXT_DETECTION"
            else: 
                raise ValueError("Invalid variable in 'mode' in request_cloud_vision_api().")
            
            api_url: str = self.GCP_CLOUD_VISION_API_URL + self.GCP_API_KEY
            headers: dict = {
                "Cotent-Type": "application/json"
            }
            data: str = json.dumps({
                "requests": [{
                    "image": {
                        "content": img_base64.decode("utf-8")
                    },
                    "features": [{
                        "type": detect_type,
                        "maxResults": 10
                    }]
                }]
            })

            res: requests.models.Response = requests.post(api_url, headers=headers, data=data)
            return res.json()
        except Exception as e:
            traceback.print_exc()

    def get_paragraphs_boundingBox(self, res: dict) -> (np.ndarray, np.ndarray):
        '''
        paragraphsの取得を，get_paragraphs()のような単純な処理ではなく，各wordsの
        sysmbolsの加算によって得ることで，フレームの頂点座標とセットで取得できる
        ようにする．

        ※paragraphsレベルの終了，または改行が直後に来るとdetect("detectedBreak")される
        symbolが来たらループを抜けてテキスト・座標の配列追加，文字列の初期化してループに戻る流れ．

        Args:
            res(dict): Vision APIからのレスポンス。
        Returns:
            texts(np.ndarray): 検出テキストのnumpy配列。
            frame_list(np.ndarray): 頂点座標集合の2次元numpy配列。
        Hints:
            ●paragraphs(x, y)
            ・res["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][for X]["paragraphs"][0]["boundingBox"]["vertices"][for X]["x"|"y"]
            ・上と下は同等の情報を持つ
            ・res["response"][0]["fullTextAnnotation"]["pages"][0]["blocks"][for X]["boundingBox"]["vertices"][for X]["x"|"y"]
            ●detectedBreak
            ・res["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][for X]["paragraphs"][0]["words"][for X]["symbols"][for X]["properties"]["detectedBreak"]["type"]
            ●symbols
            ・res["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][for X]["paragraphs"][0]["words"][for X]["symbols"][for X]["text"]
        '''
        texts: np.ndarray = np.array(["null"])
        frame_list: np.ndarray = np.array([[[0, 0], [0, 0], [0, 0], [0, 0]]])
        blocks: dict = res["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"]
        pts: list = []
        for b in blocks:
            try: # できれば無しにしたいが…
                vertices: dict = b["boundingBox"]["vertices"]
                for v in vertices:
                    pts.append([v["x"], v["y"]])
            except (KeyError, ValueError):
                pts.clear()
                continue

            for p in b["paragraphs"]:
                paragraph = ""
                for w in p["words"]:
                    for s in w["symbols"]:
                        paragraph += s["text"]
                        # print(paragraph)
                        if "property" in s:
                            if "detectedBreak" in s["property"]:
                                if s["property"]["detectedBreak"]["type"] == "SPACE":
                                    paragraph += " "
                                if s["property"]["detectedBreak"]["type"] == "LINE_BREAK":
                                    texts = np.append(texts, [paragraph], axis=0)
                                    frame_list = np.append(frame_list, [pts], axis=0)
                                    paragraph = "" # これを最後にfor sを抜けるけど，念の為
                                if s["property"]["detectedBreak"]["type"] == "EOL_SURE_SPACE":
                                    texts = np.append(texts, [paragraph], axis=0)
                                    frame_list = np.append(frame_list, [pts], axis=0)
                                    paragraph = ""
            pts.clear()

        # 初期化のためのダミーを除去
        texts = np.delete(texts, 0, axis=0)
        frame_list = np.delete(frame_list, 0, axis=0)
        return texts, frame_list

    def get_similar_paragraphs_boundingBox(self, res: dict, usr_keyword: str, min_ratio: float = 0.55) -> (np.ndarray, np.ndarray):
        '''
        「get_paragraphs_boundingBox() + extract_by_similarity_to_usr_keyword()」の関数。

        Args:
            res(dict): Vision APIからのレスポンス。
            usr_keyword(str): ユーザ入力。
            min_ratio(float): 類似度の閾値。
        Returns:
            texts(np.ndarray): 検出テキストのnumpy配列。
            frame_list(np.ndarray): 頂点座標集合の2次元numpy配列。
        Hints:
            ●paragraphs(x, y)
            ・res["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][for X]["paragraphs"][0]["boundingBox"]["vertices"][for X]["x"|"y"]
            ・上と下は同等の情報を持つ
            ・res["response"][0]["fullTextAnnotation"]["pages"][0]["blocks"][for X]["boundingBox"]["vertices][for X]["x"|"y"]
            ●detectedBreak
            ・res["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][for X]["paragraphs"][0]["words"][for X]["symbols"][for X]["properties"]["detectedBreak"]["type"]
            ●symbols
            ・res["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][for X]["paragraphs"][0]["words"][for X]["symbols"][for X]["text"]
        '''
        texts: np.ndarray = np.array(["null"])
        frame_list: np.ndarray = np.array([[[0, 0], [0, 0], [0, 0], [0, 0]]])
        blocks: dict = res["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"]
        pts: list = []
        for b in blocks:
            try: # できれば無しにしたいが…
                vertices: dict = b["boundingBox"]["vertices"]
                for v in vertices:
                    pts.append([v["x"], v["y"]])
            except (KeyError, ValueError):
                pts.clear()
                continue

            for p in b["paragraphs"]:
                paragraph = ""
                for w in p["words"]:
                    for s in w["symbols"]:
                        paragraph += s["text"]
                        # print(paragraph)
                        if "property" in s:
                            if "detectedBreak" in s["property"]:
                                if s["property"]["detectedBreak"]["type"] == "SPACE":
                                    paragraph += " "
                                if s["property"]["detectedBreak"]["type"] == "LINE_BREAK":
                                    if self.get_similarity(usr_keyword, paragraph) >= min_ratio:
                                        texts = np.append(texts, [paragraph], axis=0)
                                        frame_list = np.append(frame_list, [pts], axis=0)
                                        paragraph = "" # これを最後にfor sを抜けるけど，念の為
                                if s["property"]["detectedBreak"]["type"] == "EOL_SURE_SPACE":
                                    if self.get_similarity(usr_keyword, paragraph) >= min_ratio:
                                        texts = np.append(texts, [paragraph], axis=0)
                                        frame_list = np.append(frame_list, [pts], axis=0)
                                        paragraph = ""
            pts.clear()

        # 初期化のためのダミーを除去
        texts = np.delete(texts, 0, axis=0)
        frame_list = np.delete(frame_list, 0, axis=0)
        return texts, frame_list

    def get_paragraphs(self, res: dict) -> np.ndarray:
        '''
        Vision APIからのレスポンスから検出テキスト(Paragraphs)を「簡単に」抽出する。
        ※WordsのテキストはJSONに無さそうだったので取得は厳しそう。

        Args:
            res(dict): Vision APIからのレスポンス。
        Returns:
            paragraphs(np.ndarray): 検出テキスト(Paragraphs)のnumpy配列。
        Hints:
            ・res["responses"][0]["textAnnotations"][0]["description"]
            ・res["responses"][0]["fullTextAnnotation"]["text"]
        '''
        paragraphs = np.array(res["responses"][0]["fullTextAnnotation"]["text"].split("\n"))
        return paragraphs

    def get_boundingBox(self, res: dict, mode: str = "p") -> np.ndarray:
        '''
        Vision APIからのレスポンスからフレームの頂点座標を抽出する。

        Args:
            res(dict): Vision APIからのレスポンス。
            mode(str): フレームの粒度の選択。p=paragraphs(1単語以上の組み合わせ・フレーズごと)、w=words(単語ごと)。
        Returns:
            frame_list(np.ndarray): 頂点座標集合の2次元numpy配列。
        Hints:
            ●paragraphs
            ・res["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][for X]["paragraphs"][0]["boundingBox"]["vertices"][for X]["x"|"y"]
            ・上と下は同等の情報を持つ
            ・res["response"][0]["fullTextAnnotation"]["pages"][0]["blocks"][for X]["boundingBox"]["vertices][for X]["x"|"y"]

            ●words
            ・res["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"][for X]["paragraphs"][0]["words"][for X]["boundingBox"]["vertices"][for X]["x"|"y"]
        '''
        frame_list: np.ndarray = np.array([[[0, 0], [0, 0], [0, 0], [0, 0]]])
        blocks: dict = res["responses"][0]["fullTextAnnotation"]["pages"][0]["blocks"]
        pts: list = []

        if mode == "p":
            for b in blocks:
                try:
                    vertices: dict = b["boundingBox"]["vertices"]
                    for v in vertices:
                        pts.append([v["x"], v["y"]])
                    frame_list = np.append(frame_list, [pts], axis=0)
                    pts.clear()
                except (KeyError, ValueError):
                    continue

        if mode == "w":
            for b in blocks:
                words: dict = b["paragraphs"][0]["words"]
                for w in words:
                    try:
                        vertices: dict = w["boundingBox"]["vertices"]
                        for v in vertices:
                            pts.append([v["x"], v["y"]])
                        frame_list = np.append(frame_list, [pts], axis=0)
                        pts.clear()
                    except (KeyError, ValueError):
                        continue

        # 初期化のためのダミーを除去
        frame_list = np.delete(frame_list, 0, axis=0)
        return frame_list

    def extract_by_similarity_to_usr_keyword(self, texts: np.ndarray, frame_list: np.ndarray, usr_keyword: str, min_ratio: float = 0.55) -> (np.ndarray, np.ndarray):
        '''
        usr_keywordとtextsとの類似度判定をして、類似度が低い場合にdetected_texts・frame_lists
        からnp.delete()していく。
        ※textsとframe_listの整合性を取るため，直前にget_paragraphs_boundingBox()を使用するこ
        とが必至。

        Args:
            texts(np.ndarray): 検出テキストのnumpy配列。
            frame_list(np.ndarray): 頂点座標集合の2次元numpy配列。
            usr_keyword(str): ユーザ入力のテキスト。
            min_ratio(float): 類似度の閾値。
        Returns:
            texts(np.ndarray): 検出テキストのnumpy配列。
            frame_list(np.ndarray): 頂点座標集合の2次元numpy配列。
        '''
        try:
            if len(texts) != len(frame_list):
                raise IndexError("Different size of args between 'texts' and 'frame_list' in 'extract_by_similarity_to_usr_keyword()'.")

            del_cnt: int = 0
            curr_i: int = 0
            for i in range(len(texts)):
                curr_i = i - del_cnt
                if self.get_similarity(usr_keyword, texts[curr_i]) < min_ratio:
                    texts = np.delete(texts, curr_i)
                    frame_list = np.delete(frame_list, curr_i, axis=0)
                    del_cnt += 1

            return texts, frame_list
        except IndexError:
            traceback.print_exc()

    def get_polyframed_img(self, polyframe_list: np.ndarray, filepath: str = "", img: np.ndarray = None, color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
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

    def get_rectframed_img(self, rectframe_list: np.ndarray, filepath: str = "", img: np.ndarray = None, color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
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
                raise ValueError("'get_rectframed_img()' requires filepath OR img.")
            
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

    def get_similarity(self, usr_text: str, detected_text: str) -> float:
        '''
        検出されたテキストがユーザ入力のテキストとどれだけ類似しているかを算出する。

        とりま簡易に実装。部分文字列の類似度も一応できる。

        Args:
            usr_text(str): ユーザ入力のテキスト。
            detected_text(str): 検出されたテキスト。
        Returns:
            ratio(float): 類似度。
        '''
        shoter_text = usr_text if len(usr_text) <= len(detected_text) else detected_text
        longer_text = detected_text if shoter_text == usr_text else usr_text
        shoter_len = len(shoter_text)
        longer_len = len(longer_text)
        
        ratio: float = max([SequenceMatcher(None, shoter_text, longer_text[i:i+shoter_len]).ratio() for i in range(longer_len - shoter_len + 1)])

        if not usr_text or not detected_text: # ゴリゴリ
            ratio = 0.0
        return ratio