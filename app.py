from flask import Flask, render_template
import json

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def login_detail():
    return render_template('index.html')


@app.route('/subscription', methods=['GET', 'POST'])
def subscription():
    return render_template('subscription.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    return render_template('register.html')


@app.route('/index1', methods=['GET', 'POST'])
def index():
    
    result_data = json.load(open('Output/result_data.json', 'r'))
    sentiments = json.load(open('Output/news_sentiment.json', 'r'))

    return render_template('index1.html', result_data=result_data, sentiments=sentiments)


@app.route('/TCS', methods=['GET', 'POST'])
def TCS_tech():
    return render_template('Tcs.html')

# @app.route('/HDFC', methods=['GET', 'POST'])
# def HDFC():
#     return render_template('.html')

# @app.route('/L&T', methods=['GET', 'POST'])
# def LNT():
#     return render_template('tcs.html')

# @app.route('/TITAN', methods=['GET', 'POST'])
# def TITAN():
#     return render_template('tcs.html')

if __name__ == '__main__':
    app.run()