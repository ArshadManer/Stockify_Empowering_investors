from flask import Flask, render_template
import json
from graph import TechnicalAnalysis
import pickle
data_path = pickle.load(open('Tech_data_path.pkl', 'rb'))

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
    # file_paths = ["C:/Users/arsha/OneDrive/Desktop/TCS_data.csv"]
    file_paths = [file for file in data_path if file.endswith('TCS_data.csv')]
    ta = TechnicalAnalysis(file_paths)
    
    fig1 = ta.open_close(file_paths[0])
    graph_div1 = fig1.to_html(full_html=False)
    
    fig2 = ta.calculate_sma_signals_plot(file_paths[0])
    graph_div2 = fig2.to_html(full_html=False)
    
    fig3 = ta.calculate_ema_signals_plot(file_paths[0])
    graph_div3 = fig3.to_html(full_html=False)
    return render_template('tcs.html',graph_div1=graph_div1,graph_div2=graph_div2,graph_div3=graph_div3)


if __name__ == '__main__':
    app.run(host = "0.0.0.0",port = 8080)