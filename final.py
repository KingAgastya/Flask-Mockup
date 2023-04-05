from flask import Flask, jsonify, request
import csv

all_articles = []
with open('articles.csv', encoding="utf8") as f:
    reader = csv.reader(f)

    data = list(reader)
    all_articles = data[1:]

liked_articles = []
not_liked_articles = []
did_not_read = []

app = Flask(__name__)

@app.route("/get-article")

def get_article():
    return jsonify({
        "data" : all_articles[0],
        "status" : "success"
    })

@app.route("/liked-article", methods = ["POST"])

def liked_article():
    article = all_articles[0]
    all_articles = all_articles[1:]

    liked_articles.append(article)

    return jsonify({
        "status" : "success"
    }),
    201

@app.route("/unliked-article", methods = ["POST"])

def unliked_article():
    article = all_articles[0]
    all_articles = all_articles[1:]

    not_liked_articles.append(article)

    return jsonify({
        "status" : "success"
    }),
    201

@app.route("/did-not-read", methods = ["POST"])

def did_not_read():
    article = all_articles[0]
    all_articles = all_articles[1:]

    did_not_read.append(article)

    return jsonify({
        "status" : "success"
    }),
    201

if __name__ == "__main__":
    app.run()
