import json
import urllib.request

# url = "http://127.0.0.1:5000/"
url = <REST API URL>


# test
# req = urllib.request.Request(url)
# with urllib.request.urlopen(req) as res:
#     print(res.read().decode("utf-8"))

# GET
# get_url = url + "get_id?id=5"
# get_url = url + "get_post_id/5"
# req = urllib.request.Request(get_url)
# with urllib.request.urlopen(req) as res:
#     print(res.read().decode("utf-8"))



# POST
post_url = url + "post_json"
data = {
    "greet": "Здравствуйте!",
    "name": "Неко Дзиро"
}
headers = {
    "Content-Type": "application/json"
}
req = urllib.request.Request(post_url, json.dumps(data).encode(), headers)
with urllib.request.urlopen(req) as res:
    print(res.read().decode("utf-8"))