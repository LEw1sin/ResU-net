from flask import Flask, jsonify, request,render_template,url_for,redirect,make_response
import os
from predict2 import predict,calculate_ejection,calculate_stroke_volume,get_gender,get_age,get_name
import zipfile
import io
from static.api.Baidu import picture_enhance
import numpy as np
import threading
app = Flask(__name__)
ROOT_PATH = ''
RESULT_PATH = './results'
operation_status = {}
results = {}

def get_result(input_path, output_path):
    result_V = predict(input_path, 'label','static/result') * 1.7
    stroke_volume = calculate_stroke_volume(result_V)
    ejection = calculate_ejection(stroke_volume, result_V)
    # ndarray转list
    result_V = result_V.reshape(1,25).tolist() * 8
    result_path = 'static/result'
    return result_path, result_V, ejection, stroke_volume
    
def get_patient_info(input_path):  
    input_list = os.listdir(input_path)
    input_path = os.path.join(input_path, input_list[0])
    patient_name = get_name(input_path)
    patient_gender = get_gender(input_path)
    patient_age = get_age(input_path)
    return patient_name, patient_gender, patient_age


def mkdir(name):
    if os.path.exists(name):
        return 'existing dir ' + name
    else:
        os.makedirs(name=name)
        return 'make dir ' + name

# 默认路由
@app.route('/')
def index():
    return render_template("index.html")

# 默认路由上传方法
@app.route('/',methods=['POST'])
def file_upload():
    file_ = request.files['file']
    print(file_)
    
    foldername = os.path.splitext(file_.filename)
    
    print(foldername)
    if(foldername[0] == None or foldername[1] != '.zip'):
        return 'Wrong file type!'
    zipfile_data = io.BytesIO(file_.read())
    with zipfile.ZipFile(zipfile_data,'r') as zip_ref:
        zip_ref.extractall(os.path.join(os.getcwd(),'static/uploads'))
        
    operation_status[foldername[0]] = 'processing'
    threading.Thread(target=process_file, args=(foldername[0],)).start()
    return jsonify({'redirect':'/loading/%3F'+foldername[0]})

def process_file(foldername):
    input_path = os.path.join(os.getcwd(), 'static/uploads', foldername)
    output_path = os.path.join(os.getcwd(), 'static/result', foldername)
    input_list = os.listdir(input_path)
    input_list = os.path.join(input_path, input_list[0])
    patient_name,patient_gender,patient_age = get_patient_info(input_list)
    result_path, result_V, ejection, stroke_volume= get_result(input_path, output_path)
    folder_list = os.listdir(output_path)
    results[foldername] = {'input_path': input_path, 'result_path': result_path, 'result_V': result_V,
                           'folder_list': folder_list, 'ejection': ejection, 'stroke_volume': stroke_volume,
                           'patient_name': patient_name, 'patient_gender': patient_gender, 'patient_age': patient_age}
    operation_status[foldername] = 'done'


@app.route('/loading/?<string:foldername>')
def loading(foldername):
    return render_template("loading.html",foldername=foldername)

@app.route('/check_status/<foldername>')
def check_status(foldername):
    status = operation_status.get(foldername)
    if status == None:
        status = 'error'
    return jsonify({'status': status, 'foldername': foldername})



# 结果页面路由
@app.route('/about/?<string:foldername>')
def index_get(foldername):
    return render_template("about_tmp.html", input_path=results[foldername]['input_path'], result_path=results[foldername]['result_path'], 
                           result_V=results[foldername]['result_V'], folder_list=results[foldername]['folder_list'], ejection=results[foldername]['ejection'], 
                           stroke_volume=results[foldername]['stroke_volume'], patient_name=results[foldername]['patient_name'] ,
                           patient_age=results[foldername]['patient_age'] ,patient_gender=results[foldername]['patient_gender'] ,foldername=foldername)

# 图像增强操作
@app.route('/about/?<string:foldername>',methods=['POST'])
def enhance_picture(foldername):
    input_path = request.form.get('input_path')
    # 此处调用图像增强函数
    picture_enhance(input_path)
    # 返回带有结果的网页
    return redirect(url_for('index_get',foldername=foldername))
    
    
@app.route('/get_images',methods=['GET'])
def get_images():
    # 此处需要结果目录
    foldername = request.args.get('path')
    foldername = int(foldername)
    name = request.args.get('name')
    folder_path = 'static/result'  
    folder_path = os.path.join(os.getcwd(),folder_path,name)
    foldername = sorted(os.listdir(folder_path))[foldername]
    folder_path = os.path.join(folder_path,foldername)
    print(os.getcwd())
    images = []
    image_extensions = ('.png', '.jpg', '.jpeg')
    for root, dirs, files in os.walk(folder_path):
        # if root == folder_path:  # 判断当前是否为根目录
        for file in files:
            if file.lower().endswith(image_extensions):
                images.append(os.path.join('../static/result',name,foldername,file))
                print(os.path.join('../static/result',name,foldername,file))
    '''  else:
        # 如果不是根目录，则移除子文件夹路径以避免遍历子文件夹内的图片
        dirs[:] = []'''
    print(images)
    # 输出当前目录下的图片路径
    return jsonify({'images': images})

@app.route('/te1')
def te1():
    return render_template('index_new.html')

@app.route('/get_folders', methods=['POST'])
def get_folders():
    path = 'static/uploads'
    # 获取指定路径下的文件夹列表
    folder_list = []
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            folder_list.append(item)
    return jsonify({'folders': folder_list})


# 输出当前目录下的图片路径

@app.route('/get_folder_data', methods=['GET'])
def get_folder_data():
    folder_path = request.args.get('folderPath')
    # 检查文件夹路径是否存在
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return jsonify([])

    # 获取文件夹中的内容列表
    content_list = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        item_type = 'file' if os.path.isfile(item_path) else 'folder'
        content_list.append({'name': item, 'path': item_path, 'type': item_type})

    return jsonify(content_list)





if __name__ == '__main__':
    app.debug = True
    app.run(port=7777)
    
