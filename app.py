from flask import Flask, render_template
import json
from graph import TechnicalAnalysis
import os

file_paths = [os.path.join(os.getcwd(), "raw_data", file) for file in os.listdir("raw_data")]

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def login_detail():
    return render_template('login.html')


@app.route('/subscription', methods=['GET', 'POST'])
def subscription():
    return render_template('subscription.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    return render_template('register.html')



@app.route('/dashboard', methods=['GET', 'POST'])
def index():
    
    result_data = json.load(open('Output/result_data.json', 'r'))
    sentiments = json.load(open('Output/news_sentiment.json', 'r'))

    return render_template('dashboard.html', result_data=result_data, sentiments=sentiments)

#1
@app.route('/TCS', methods=['GET', 'POST'])
def TCS_tech():
    
    file_path = [file for file in file_paths if file.endswith('TCS_data.csv')]
    ta = TechnicalAnalysis(file_path)
    
    fig1 = ta.open_close(file_path[0])
    graph_div1 = fig1.to_html(full_html=False)
    
    fig2 = ta.calculate_sma_signals_plot(file_path[0])
    graph_div2 = fig2.to_html(full_html=True)
    
    fig3 = ta.calculate_ema_signals_plot(file_path[0])
    graph_div3 = fig3.to_html(full_html=True)
    
    fig4 = ta.analyze_rsi(file_path[0])
    graph_div4 = fig4.to_html(full_html=True)
    
    return render_template('TCS.html',graph_div1=graph_div1, graph_div2=graph_div2, graph_div3=graph_div3, graph_div4=graph_div4)

#2
@app.route('/HDFC', methods=['GET', 'POST'])
def HDFC_tech():
    
    file_path = [file for file in file_paths if file.endswith('HDFCBANK_data.csv')]
    ta = TechnicalAnalysis(file_path)
    
    fig1 = ta.open_close(file_path[0])
    graph_div1 = fig1.to_html(full_html=False)
    
    fig2 = ta.calculate_sma_signals_plot(file_path[0])
    graph_div2 = fig2.to_html(full_html=True)
    
    fig3 = ta.calculate_ema_signals_plot(file_path[0])
    graph_div3 = fig3.to_html(full_html=True)
    
    fig4 = ta.analyze_rsi(file_path[0])
    graph_div4 = fig4.to_html(full_html=True)
    
    return render_template('HDFC.html',graph_div1=graph_div1, graph_div2=graph_div2, graph_div3=graph_div3, graph_div4=graph_div4)

@app.route('/L&T', methods=['GET', 'POST'])
def LNT_tech():
    
    file_path = [file for file in file_paths if file.endswith('LT_data.csv')]
    ta = TechnicalAnalysis(file_path)
    
    fig1 = ta.open_close(file_path[0])
    graph_div1 = fig1.to_html(full_html=False)
    
    fig2 = ta.calculate_sma_signals_plot(file_path[0])
    graph_div2 = fig2.to_html(full_html=True)
    
    fig3 = ta.calculate_ema_signals_plot(file_path[0])
    graph_div3 = fig3.to_html(full_html=True)
    
    fig4 = ta.analyze_rsi(file_path[0])
    graph_div4 = fig4.to_html(full_html=True)
    
    return render_template('L&T.html',graph_div1=graph_div1, graph_div2=graph_div2, graph_div3=graph_div3, graph_div4=graph_div4)

@app.route('/TITAN', methods=['GET', 'POST'])
def TITAN_tech():
    
    file_path = [file for file in file_paths if file.endswith('TITAN_data.csv')]
    ta = TechnicalAnalysis(file_path)
    
    fig1 = ta.open_close(file_path[0])
    graph_div1 = fig1.to_html(full_html=False)
    
    fig2 = ta.calculate_sma_signals_plot(file_path[0])
    graph_div2 = fig2.to_html(full_html=True)
    
    fig3 = ta.calculate_ema_signals_plot(file_path[0])
    graph_div3 = fig3.to_html(full_html=True)
    
    fig4 = ta.analyze_rsi(file_path[0])
    graph_div4 = fig4.to_html(full_html=True)
    
    return render_template('TITAN.html',graph_div1=graph_div1, graph_div2=graph_div2, graph_div3=graph_div3, graph_div4=graph_div4)


if __name__ == '__main__':
    app.run(host = "0.0.0.0",port = 8080)