from flask import Flask, flash, render_template, make_response, request, session, redirect, url_for, abort
from werkzeug.utils import secure_filename
import urllib, json
from datetime import datetime
import os
import inference
import cv2
from numba import cuda
import numpy as np

UPLOAD_FOLDER = 'static/upload/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.secret_key = '\x8b\x19\xa1\xb0D\x87?\xc1M\x04\xff\xc8\xbdE\xb1\xca\xe6\x9e\x8d\xb3+\xbe>\xd2'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Main Page
@app.route('/')
def index():
    return make_response(open('index.html').read())
    """
    ip_user = request.remote_addr
    loc_api = 'http://ip-api.com/json/' + ip_user
    try:
        loc_info = urllib.request.urlopen(loc_api).read()
        loc_json = json.loads(loc_info)
        loc_city = loc_json['city']
        loc_country = loc_json['country']
        loc_isp = loc_json['isp']

        if loc_country == 'United Kingdom' or "China":
            f_ip = open('ip_check.txt', mode='a')
            f_ip.write(
                str(datetime.now()) + ', ' + loc_city + ', ' + loc_country + ', ' + loc_isp + ', ' + ip_user + '\n')
            f_ip.close()
            return make_response(open('index.html').read())
        else:
            return abort(403)
    except:
        # f_ip=open('ip_check.txt',mode='a')
        # f_ip.write(str(datetime.now())+', '+ip_user+' cannot be checked'+'\n')
        # f_ip.close()
        return abort(403)
    """


@app.route('/bg_removal', methods=['POST'])
def clicked():
    file = request.files['file']
    if file:
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_path = UPLOAD_FOLDER+filename
        else:
            flash('Uploaded image type is NOT valid!')
            return make_response(open('index.html').read())
    else:
        flash('Neither file nor url given for image removal !')
        return make_response(open('index.html').read())

    choice = request.form['choice']
    choice = eval(choice)
    print(choice)
    labels, new = inference.matting(file_path, choice)

    #!!!!!!!!!!!!!!!
    if len(np.unique(labels))==1:
        print('failed to get segmentation!')
        flash('No person detected in the image !')
        return make_response(open('index.html').read())

    #mat2 = inference.mat_2(file_path, mat1)
    out_path = os.path.join('static/download/', os.path.splitext(os.path.basename(filename))[0]) + '-AIT.png'
    return render_template('show.html', img_path=out_path)


@app.route('/robots.txt', methods=['GET'])
def robots():
    response = make_response(open('robots.txt').read())
    response.headers["Content-type"] = "text/plain"
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=80)