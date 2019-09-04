import re
import traceback
from flask import Flask, jsonify, make_response, request
from modules.spine_detect import SpineDetect
from modules.async_process import AsyncAPIRequest, AsyncAPIRequest2

sd = SpineDetect()
async_process = AsyncAPIRequest()
async_process2 = AsyncAPIRequest2()

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False # 日本語表記対応

# curl localhost:5000
@app.route("/")
def test():
    return jsonify({"res": "Hello, NekoJiro!"})

# curl localhost:5000/get_id?id=<int:id>
@app.route("/get_id", methods=["GET"])
def ex_get_id():
    params: dict = request.args
    try:
        if not re.compile(r'^[0-9]+$').match(params["id"]) or int(params["id"]) < 0:
            # 0以上の整数でなければエラー応答する
            raise ValueError("Invalid data.")
        res: dict = {
            "id": params["id"],
            "name": "地球規模",
            "call": "0000-000-000",
            "latitude": 60,
            "longitude": 130,
            "menus": {
                "ラーメン": 900,
                "汁なし": 900
            }
        }
        return make_response(jsonify(res))
    except Exception as e:
        result = error_handler(e)
        return result
        
# curl -X GET localhost:5000/get_post_id/<int:id>
# curl -X POST localhost:5000/get_post_id/<int:id>
@app.route("/get_post_id/<int:id>", methods=["GET", "POST"])
def ex_get_post_id(id):
    res: dict = {
        "id": id,
        "name": "地球規模",
        "call": "0000-000-000",
        "latitude": 60,
        "longitude": 130,
        "menus": {
            "ラーメン": 900,
            "汁なし": 900
        }
    }
    return make_response(jsonify(res))

# curl -X POST -H 'Content-Type:application/json' -d '{"greet": "Здравствуйте!", "name": "Неко Дзиро"}' localhost:5000/post_json
@app.route("/post_json", methods=["POST"])
def ex_post_json():
    try:
        req: dict = request.json # .get_json()
        # print("============[req]============")
        # print(req)
        res: dict = {
            "message": "{name}, {greet}".format(name=req["name"], greet=req["greet"])
        }
        return make_response(jsonify(res))
    except Exception as e:
        # print("============[e]============")
        # print(e)
        result = error_handler(e)
        return result

@app.route("/test_img", methods=["POST"])
def test_img():
    try:
        req: dict = request.json
        print(req)
        img_base64: bytes = req["image"] # .encode("utf-8")
        img: np.ndarray = sd.img_base642np(img_base64)
        print(img)
        print(type(img))
        img_base64 = sd.img_np2base64(img)
        res = {
            "image": img_base64.decode("utf-8")
        }
        return make_response(jsonify(res))
    except Exception as e:
        traceback.print_exc()
        result = error_handler(e)
        return result


@app.route("/honaru", methods=["POST"])
def honaru():
    try:
        isFirst: str = request.headers.get("isFirst", "0") # ヘッダーからアプリ起動直後かのフラグ取得 # デフォルト値1の方がいいかも
        req: dict = request.json

        if int(isFirst):
            async_process.initialize()
        '''
        同期処理の場合
        '''
        '''
        usr_keyword: str = req["request"]["keyword"]
        img_base64: bytes|int = req["request"]["image"] # .encode("utf-8") # int(0)
        # (1)
        img: np.ndarray = sd.img_base642np(img_base64)
        # (2)
        scale_rate = 1.0
        new_img: np.ndarray = sd.img_preprocess(img, scale_rate=scale_rate)
        # (3) # (3)と(4)は非同期化または並列化して高速化の見込みありだが実装の時間なし
        result: dict = sd.request_cloud_vision_api(sd.img_np2base64(new_img), mode="doc")
        if len(result["responses"][0]) > 0: # 画像の質悪によるレスポンスが空かのチェック
            # (5)
            texts, frame_list = sd.get_paragraphs_boundingBox(result) # texts: np.ndarray, frame_list: np.ndarray
            _, frame_list = sd.extract_by_similarity_to_usr_keyword(texts, frame_list, usr_keyword, min_ratio=0.55)
            # (6)
            # frame_list = frame_list / scale_rate
            img = sd.get_polyframed_img(frame_list, img=img, thickness=4) # sd.get_rectframed_img(frame_list, img=img)
            # (8)
            img_base64 = sd.img_np2base64(img)
        else:
            img_base64 = 0
        # (4)
        itemList: list = ya.itemList(keyword)
        '''


        '''
        非同期処理の場合
        '''
        img_base64, itemList = async_process.async_api_req(req) # img_base4: bytes|int, itemList: list|int
        
        # (9)
        data: dict = {
            "image": img_base64.decode("utf-8") if img_base64 else str(img_base64),
            "itemList": itemList if itemList != -1 else str(itemList)
        }
        res = make_response(jsonify(data))
        res.headers["serverID"] = 2 # Server ID．app(master)が1，test(develop)が2，app2(master2)が3

        return res
    except Exception as e:
        traceback.print_exc()
        result = error_handler(e)
        return result


@app.route("/honaru2", methods=["POST"])
def honaru2():
    try:
        isFirst: str = request.headers.get("isFirst", "0") # ヘッダーからアプリ起動直後かのフラグ取得 # デフォルト値1の方がいいかも
        req: dict = request.json

        if int(isFirst):
            async_process2.initialize()

        frame_list, itemList = async_process2.async_api_req(req) # frame_list: list|int, itemList: list|int
        
        # (9)
        data: dict = {
            "image": frame_list if frame_list else str(frame_list),
            "itemList": itemList if itemList != -1 else str(itemList)
        }
        res = make_response(jsonify(data))
        res.headers["serverID"] = 2 # Server ID．app(master)が1，test(develop)が2，app2(master2)が3

        return res
    except Exception as e:
        traceback.print_exc()
        result = error_handler(e)
        return result

# 通信エラー処理
@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(405)
@app.errorhandler(500)
def error_handler(e):
    print(e)
    res: dict = {
        "error": {
            "type": e.name, # ???
            "message": e.description
        }
    }
    return jsonify(res), e.code


if __name__ == "__main__":
    app.debug = True
    # When run on local, as same as below
    # app.run(host="127.0.0.1", port=5000)
    # Run on Heroku
    app.run()