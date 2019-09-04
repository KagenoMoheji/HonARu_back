import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import os
import requests


class Yahoo(object):#Shoppingが継承する基本的なAPI取得用のクラス
    response = None

    def __init__(self, appid=None,jan13=None):
        self.params = {
            "appid": appid,
            "jan13":jan13#?
        }

    # APIを叩く
    def request(self, url, **kwargs):
        self.params.update(kwargs)
        query_string = urllib.parse.urlencode(self.params)

        if query_string:
            url = url + query_string + "&module=priceranges&category_id=10002&hits=10"#書籍にカテゴリを限定してある

        with urllib.request.urlopen(url) as f:
            self.response = f.read()
            self.request_url = f.geturl()
        return self

    # XMLをパースしたデータ
    @property
    def parse(self):
        if self.response is None:
            raise
        return ET.fromstring(self.response)



class Shopping(Yahoo):#Yahooショッピングを使うためのクラス．キーワードを受け取ってURL，JanCode,レビュー平均を返してくれる

    NS = {
        "itemSearch": "urn:yahoo:jp:itemSearch"
        # "itemLookup": "urn:yahoo:jp:itemLookup"
    }

    def search(self, **kwargs):
        self.prefix = "itemSearch"
        url = "http://shopping.yahooapis.jp/ShoppingWebService/V1/itemSearch?"
        return self.request(url, **kwargs)

    # 商品情報を取得するジェネレータ
    @property
    def items(self):
        match = "{}:Hit".format(self.prefix)
        for item in self.parse[0].findall(match, self.NS):
            self._item = item
            yield self

    def find(self, tag):
        item = self._item
        for t in tag.split("."):
            item = item.find("{}:{}".format(self.prefix, t), self.NS)
        return item

    @property
    def name(self):
        return self.find("Name").text

    @property
    def url(self):
        return self.find("Url").text

    @property
    def jancode(self):
        return self.find("JanCode").text

    # @property
    # def rate(self):
    #     return self.find("Review.Rate").text



class BookSearch:#書籍検索
    def __init__(self,word): # コンストラクタでインスタンスを初期化
        APPID = os.environ["Y_API_KEY"]
        self.shopping = Shopping(APPID)
        self.response = self.shopping.search(**{"query":word})

    def oi(self):
        oi = 1
        try:
            for item in self.response.items:
                n = item.find("JanCode").text
        except AttributeError:
            print("検索結果は0件です")
            oi = 0
        return oi

    def Isbn_Name(self):
        Isbn_name_dic = {}
        try:
            for item in self.response.items:
                if item.find("JanCode").text != None:
                    Isbn_name_dic[item.find("JanCode").text] = item.find("Name").text
        except AttributeError:
            print("検索結果がありませんでした")
        return Isbn_name_dic



    def Isbn_Rev(self):
        Isbn_rev_dic = {}
        try:
            for item in self.response.items:
                if item.find("Review.Rate").text != None:
                    Isbn_rev_dic[item.find("JanCode").text] = float(0)
                    if float(item.find("Review.Rate").text) > Isbn_rev_dic[item.find("JanCode").text]:
                        Isbn_rev_dic[item.find("JanCode").text] = float(item.find("Review.Rate").text)
        except AttributeError:
            print("検索結果がありませんでした")
        return Isbn_rev_dic


    def Isbn_Page(self):
        Isbn_page_dic = {}
        try:
            for item in self.response.items:
                if item.find("JanCode").text != None:
                    Isbn_page_dic[item.find("JanCode").text] = "https://shopping.yahoo.co.jp/search/ISBN+"+item.find("JanCode").text+"/0/"
        except AttributeError:
            print("検索結果がありませんでした")
        return Isbn_page_dic

    def Isbn_Img(self):
        Isbn_img_dic = {}
        try:
            for item in self.response.items:
                if item.find("JanCode").text != None:
                    Isbn_img_dic[item.find("JanCode").text] = item.find("Image.Medium").text
        except AttributeError:
            print("検索結果がありませんでした")
        return Isbn_img_dic




def itemList(keyword):
    new_booksearch = BookSearch(keyword)
    itemList = []
    if new_booksearch.oi() == 1:

        for key in new_booksearch.Isbn_Name().keys():

            response = requests.get("https://api.openbd.jp/v1/get?isbn="+key)#書籍名と著者が綺麗に取れるAPI

            item_dic = {}
            try:
                data = response.json()[0]["summary"]
                item_dic["title"] = data["title"]
                item_dic["author"] = data["author"]
                del data
            except TypeError:
                item_dic["title"] = new_booksearch.Isbn_Name()[key]
                item_dic["author"] = ""
            item_dic['Review'] = Review_Average(key)
            item_dic['URL'] = new_booksearch.Isbn_Page()[key]
            item_dic['itemImage'] = new_booksearch.Isbn_Img()[key]


            itemList.append(item_dic)
    return itemList


def Review_Average(jan13):
    APPID = os.environ["Y_API_KEY"]
    url = "http://shopping.yahooapis.jp/ShoppingWebService/V1/reviewSearch?appid="+APPID+"&jan="+str(jan13)
    req = urllib.request.Request(url)

    with urllib.request.urlopen(req) as response:
        XmlData = response.read()
    new_XmlData = XmlData.decode("utf-8")
    root = ET.fromstring(new_XmlData)

    n = 0
    s = 0
    for rate in root.iter('{urn:yahoo:jp:reviewSearch}Rate'):
        n = n+1
        s = s+float(rate.text)
    print()
    if n == 0:
        return 0
    else:
        return s/n