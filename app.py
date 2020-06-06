#import thư viện
from flask import Flask, render_template, request, url_for, session
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
import uuid #Random Short Id
import os
import scipy.cluster.hierarchy as sch 
from sklearn.cluster import AgglomerativeClustering

UPLOAD_FOLDER = 'static/uploads/kmeans' #Đường dẫn thư mục upload
ALLOWED_EXTENSIONS = {'csv'}#tập tin cho phép

app = Flask(__name__)#mở đầu thư viện flask

#upload tập tin 
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' #Secret key of Session
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


#hàm truyền vào khi khởi động, hiển thị trang index chính
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')# Trang chủ 

#vào thư mục kmeans, hiển thị trang index của giải thuật kmeans
@app.route('/kmeans', methods=['GET', 'POST'])
def kmeans_index():
    return render_template('kmeans/index.html')# Trang chủ 

#File được cho phép
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Hiển thị data của giải thuật kmeans khi nạp dữ liệu
@app.route('/kmeans/data', methods=['GET', 'POST'])
def kmeans_data():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            errors = 'No file part! Please choose 1 file csv !'
            return render_template('kmeans/data.html', errors=errors)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            errors = 'No selected file'
            return render_template('kmeans/data.html', errors=errors)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4())[:8] + '_' + filename)
            file.save(file_path)

            session['csvfile'] = file_path #Save path file to session
            columns_name_attribute = ['Col1','Col2','Col3','Col4','Col5'] 
            data = pd.read_csv(file_path, names = columns_name_attribute)
        
            m = data.shape[1]
            return render_template('kmeans/data.html', data=data.to_html(table_id='myTable', classes='table table-striped', header=True, index=False), m=m)

#Hiển thị đồ thị elbow khi người dùng chọn nhóm
@app.route('/kmeans/elbow', methods=['GET', 'POST'])
def kmeans_elbow():
    file_path = session.get('csvfile')
    columns_name_attribute = ['Col1','Col2','Col3','Col4','Col5']
    data = pd.read_csv(file_path, names = columns_name_attribute)
    col = request.form.getlist('cot') #Get values of checkbox form from
    col = np.array(col)
    col1 = col[0]
    col2 = col[1]
    session['col1'] = col1 #Save column to session
    session['col2'] = col2 #Save column to session
    m = data.shape[1]
    bd = 0
    X = data.iloc[int(bd):, [int(col1), int(col2)]]
    n = data.shape[0]
    # Tiến hành gom nhóm (Elbow)
    # Chạy thuật toán KMeans với k = (1, 10)

    clusters = []
    for i in range(1, 10):
        km = KMeans(n_clusters=i).fit(X)
        clusters.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(x=list(range(1, 10)), y=clusters, ax=ax)

    ax.set_title("Đồ thị Elbow")
    ax.set_xlabel("Số lượng nhóm")
    ax.set_ylabel("Gía trị Inertia")

    image = 'static/images_kmeans/'+ str(uuid.uuid4())[:8] +'_elbow.png'
    plt.savefig(image)
    plt.clf()

    return render_template('kmeans/elbow.html', url1='/'+image)

#hiển thị kết quả
@app.route('/kmeans/ketqua', methods=['GET', 'POST'])
def kmeans_clasf():
    file_path = session.get('csvfile')#lấy đường dẫn file csv
    columns_name_attribute = ['Col1','Col2','Col3','Col4','Col5']#tên thuộc tính cột
    data = pd.read_csv(file_path,names= columns_name_attribute)
    cola = session.get("col1")#lấy giá trị cột a
    colb = session.get("col2")#lấy giá trị cột b
    bd = 0
    X = data.iloc[int(bd):, [int(cola), int(colb)]].values#dữ liệu cần phân hoạch
    k = request.form.get('cluster')#lấy giá trị k 
    km3 = KMeans(n_clusters= int(k))#chạy giải thuật kmeans
    y_means = km3.fit_predict(X)
    listcolor = ['pink','red','blue','green','yellow']#tạo danh sách màu
    color = []
    for i in range(5):
        color.append(listcolor[i])
    for i in range(int(k)):
        plt.scatter(X[y_means == i, 0], X[y_means == i, 1], s = 100, c = color[i]) 
    #tâm của mỗi nhóm
    plt.scatter(km3.cluster_centers_[:,0], km3.cluster_centers_[:, 1], s = 100, c = 'orange' , label = 'Centeroid') 
    plt.style.use('fivethirtyeight') 
    plt.title('K Means Clustering', fontsize = 20) 
    plt.xlabel('Annual Income') 
    plt.ylabel('Spending Score') 
    plt.legend() 
    plt.grid() 
    img = 'static/images_kmeans/'+ str(uuid.uuid4())[:8] +'_kq.png'
    plt.savefig(img)
    #hiên thị hình ảnh phân hoạch
    return render_template('kmeans/ketqua.html', url2='/'+img)

###################################################################giai thuat AGG
#chạy trang index trong thư mục agg
@app.route('/agg', methods=['GET', 'POST'])
def agg_index():
    return render_template('agg/index.html')# Trang chủ

#hiển thị dữ liệu khi người dùng nạp lên trang web
@app.route('/agg/data', methods=['GET', 'POST'])
def agg_data():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            errors = 'No file part! Please choose 1 file csv !'
            return render_template('agg/data.html', errors=errors)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            errors = 'No selected file'
            return render_template('agg/data.html', errors=errors)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4())[:8] + '_' + filename)
            file.save(file_path)

            session['csvfile'] = file_path #Save path file to session
            columns_name_attribute = ['Col1','Col2','Col3','Col4','Col5'] 
            data = pd.read_csv(file_path, names = columns_name_attribute)
        
            m = data.shape[1]
            return render_template('agg/data.html', data=data.to_html(table_id='myTable', classes='table table-striped', header=True, index=False), m=m)

#Hiển thị đồ thị elbow
@app.route('/agg/dendrogram', methods=['GET', 'POST'])
def agg_dendrogram():
    file_path = session.get('csvfile')
    columns_name_attribute = ['Col1','Col2','Col3','Col4','Col5']
    data = pd.read_csv(file_path, names = columns_name_attribute)
    col = request.form.getlist('cot') #Get values of checkbox form from
    col = np.array(col)
    col1 = col[0]
    col2 = col[1]
    session['col1'] = col1 #Save column to session
    session['col2'] = col2 #Save column to session
    m = data.shape[1]
    hihi = 0
    X = data.iloc[int(hihi):, [int(col1), int(col2)]]
    n = data.shape[0]

    clusters = []
    dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward')) 
    plt.title('Dendrogram') 
    plt.xlabel('Khách hàng') 
    plt.ylabel('Khoảng cách Euclidean') 
    image = 'static/images_kmeans/'+ str(uuid.uuid4())[:8] +'_ded.png'
    plt.savefig(image)
    plt.clf()

    return render_template('agg/dendrogram.html', url1='/'+image)

#hiển thị kết quả
@app.route('/agg/ketqua', methods=['GET', 'POST'])
def agg_clasf():
    file_path = session.get('csvfile')
    columns_name_attribute = ['Col1','Col2','Col3','Col4','Col5']
    data = pd.read_csv(file_path,names= columns_name_attribute)
    cola = session.get("col1")
    colb = session.get("col2")
    bd = 0
    X = data.iloc[int(bd):, [int(cola), int(colb)]].values
    k = request.form.get('cluster')
    hc = AgglomerativeClustering(n_clusters = int(k), affinity = 'euclidean', linkage = 'ward') 
    y_hc = hc.fit_predict(X) 
    listcolor = ['pink','red','blue','green','yellow']
    color = []
    for i in range(5):
        color.append(listcolor[i])
    for i in range(int(k)):
        plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1], s = 100, c = color[i])
    plt.title('AGG Clustering', fontsize = 20) 
    plt.xlabel('Annual Income') 
    plt.ylabel('Spending Score') 
    plt.legend() 
    plt.grid() 
    img = 'static/images_kmeans/'+ str(uuid.uuid4())[:8] +'_kq1.png'
    plt.savefig(img)
    #hiển thị hình ảnh được phân hoạch
    return render_template('agg/ketqua.html', url2='/'+img) 

#hamg main
if __name__ == '__main__':
    app.run(debug=True)


