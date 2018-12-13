from flask import Flask, render_template,request,redirect

import model
import json
from flask_jsonpify import jsonpify
from flask import jsonify

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
        content = model.metamovie()
        collab = model.collabmv()

        return render_template("index.html",content=content,collab=collab)

@app.route('/keyword', methods=['GET', 'POST'])
def keyword():
    if request.method == 'POST':
        title=request.form["title"]
        result = model.content_recommendation(title)
        # recommendation = [tuple(x) for x in result]
        # recomm        endation=result.to_dict()
        df_list = result.values.tolist()
        result = jsonify(df_list)
        return result    
    else:
        return render_template("index.html")

@app.route('/cf',methods=['GET', 'POST'])
def cf():
    if request.method == 'POST':
        movie_id=request.form.getlist('id')
        # title = request.form.getlist('title')
        ratings = request.form.getlist('rating')
        # return jsonify(movie_id)
        result= model.cf_recommend(movie_id,ratings)
        result=result["title"].to_json()
        return render_template("cf.html",title=result)
    else:
        return render_template("index.html")    

if __name__ == '__main__':
    app.run(debug=True)