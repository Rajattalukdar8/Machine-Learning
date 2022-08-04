from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():

    if request.method == "POST":
        f=request.files['file']
        f.save(os.path.join(app.config["UPLOAD_PATH"], f.filename))
        return render_template("after.html", msg = "File has been uploaded successfully")
    float_feat = [line.rstrip() for line in open(f"files\\{f.filename}")]
    # float_feat = [float(x) for x in request.form.values()]
    feat = [np.array(float_feat)]
    prediction = model.predict(feat)
    with open("pred.txt", "w") as f:
        f.write(str(prediction))
    # data1 = request.form['a']
    # data2 = request.form['b']
    # data3 = request.form['c']
    # data4 = request.form['d']
    # arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(feat)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)



