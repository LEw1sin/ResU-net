import requests
import base64
def picture_enhance(input_image_path):  
    request_url = "https://aip.baidubce.com/rest/2.0/image-process/v1/contrast_enhance"
    # 二进制方式打开图片文件
    f = open(input_image_path, 'rb')
    img = base64.b64encode(f.read())
    params = {"image":img}
    access_token = '24.abf3358f2525391c01af31e7170cdcb1.2592000.1697030804.282335-39174892'
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        print ("Success")
    # 将base64格式转换为png并保存
    with open(input_image_path, "wb") as f:
        f.write(base64.b64decode(response.json()['image']))