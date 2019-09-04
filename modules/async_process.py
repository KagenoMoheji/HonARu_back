import asyncio
from modules.spine_detect import SpineDetect
import modules.yahoo_api as ya

class AsyncAPIRequest:
    sd = None
    loop = None
    curr_keyword: str = ""
    def __init__(self):
        self.sd = SpineDetect()
        self.loop = asyncio.get_event_loop()
        self.curr_keyword = ""

    def initialize(self):
        self.curr_keyword = ""

    def close(self):
        self.loop.close()

    def async_api_req(self, req: dict): # -> (bytes|int, list|int)
        if "request" in req:
            req: dict = req["request"]
        keyword: str = req["keyword"]
        img_base64: bytes = req["image"]
        
        api_req_processes = asyncio.gather(
            self.async_gcp(img_base64, keyword),
            self.async_yahoo(keyword)
        )
        results = self.loop.run_until_complete(api_req_processes)

        # https://u7fa9.org/memo/HEAD/archives/2015-08/2015-08-24_2.rst
        # 上記リンクを見る限り，実行順は不定だが，戻り値は順序通り
        return results[0], results[1]

    async def async_gcp(self, img_base64: bytes, keyword: str): # -> bytes|int
        # (1)
        img: np.ndarray = self.sd.img_base642np(img_base64)
        # (2)
        scale_rate: float = 1152 / img.shape[1] # widthの比率(min:1024)
        new_img: np.ndarray = self.sd.img_preprocess(img, scale_rate=scale_rate)
        # (3)
        result: dict = self.sd.request_cloud_vision_api(self.sd.img_np2base64(new_img), mode="doc")
        if len(result["responses"][0]) > 0: # 画像の質悪によるレスポンスが空かのチェック
            # (5)
            _, frame_list = self.sd.get_similar_paragraphs_boundingBox(result, keyword, min_ratio=0.55) # texts: np.ndarray, frame_list: np.ndarray
            if len(frame_list) > 0:
                # (6)
                frame_list = frame_list / scale_rate # 座標を元のサイズの画像に合うように調整
                img = self.sd.get_polyframed_img(frame_list, img=img, thickness=4) # self.sd.get_rectframed_img(frame_list, img=img)
                # (8)
                img_base64 = self.sd.img_np2base64(img)
            else:
                img_base64 = 0 # クライアント側で取得した画像を表示してもらう
        else:
            img_base64 = 0
        return img_base64

    async def async_yahoo(self, keyword: str): #  -> list|int
        if keyword != self.curr_keyword:
            # (4)
            itemList: list = ya.itemList(keyword)
            self.curr_keyword = keyword
            return itemList
        return -1

class AsyncAPIRequest2(AsyncAPIRequest):
    def __init__(self):
        super().__init__()

    def close(self):
        self.loop.close()

    def async_api_req(self, req: dict): # -> (list|int, list|int)
        if "request" in req:
            req: dict = req["request"]
        keyword: str = req["keyword"]
        img_base64: bytes = req["image"]
        
        api_req_processes = asyncio.gather(
            self.async_gcp(img_base64, keyword),
            super().async_yahoo(keyword)
        )
        results = self.loop.run_until_complete(api_req_processes)
        return results[0], results[1]

    async def async_gcp(self, img_base64: bytes, keyword: str): # -> list
        # (1)
        img: np.ndarray = self.sd.img_base642np(img_base64)
        # (2)
        scale_rate: float = 1152 / img.shape[1] # widthの比率(min:1024)
        new_img: np.ndarray = self.sd.img_preprocess(img, scale_rate=scale_rate)
        # (3)
        result: dict = self.sd.request_cloud_vision_api(self.sd.img_np2base64(new_img), mode="doc")
        if len(result["responses"][0]) > 0: # 画像の質悪によるレスポンスが空かのチェック
            # (5)
            _, frame_list = self.sd.get_similar_paragraphs_boundingBox(result, keyword, min_ratio=0.55) # texts: np.ndarray, frame_list: np.ndarray
            if len(frame_list) > 0:
                frame_list = frame_list.tolist()
        else:
            frame_list = 0
        return frame_list