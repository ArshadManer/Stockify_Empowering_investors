from flask import Flask, render_template,redirect,request,flash, session, url_for
import json
from graph import TechnicalAnalysis
import os
import secrets
import csv
import pickle

file_paths = [os.path.join(os.getcwd(), "raw_data", file) for file in os.listdir("raw_data")]

app = Flask(__name__)

key = secrets.token_hex(12)
app.secret_key = key
USER_CSV = [file for file in file_paths if file.endswith('USER.csv')][0]

def find_user(username: str, password: str):
    try:
        with open(USER_CSV, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['username'] == username and row['password'] == password:
                    return row
        return None
    except Exception as e:
        raise e

def write_to_csv(data):
    with open(USER_CSV, mode='a', newline='') as file:
        fieldnames = ['first_name', 'last_name', 'username', 'password']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(data)
        


@app.route('/', methods=['GET', 'POST'])
def login_detail():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            user = find_user(username.lower(), password)
            if user:
                session['username'] = username
                return redirect(url_for('index'))
            else:
                flash("Invalid login credentials. Please try again.")
        except Exception as e:
            raise e
    return render_template('login.html')


@app.route('/subscription', methods=['GET', 'POST'])
def subscription():
    return render_template('subscription.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        password = request.form['password']
        data = {
            'first_name': first_name,
            'last_name': last_name,
            'username': email,
            'password': password
        }
        write_to_csv(data)
        return redirect(url_for('subscription'))
        
    return render_template('register.html')



@app.route('/Dashboard', methods=['GET', 'POST'])
def index():
    result_data = json.load(open('Output/result_data.json', 'r'))
    sentiments = json.load(open('Output/news_sentiment.json', 'r'))
    signals= json.load(open('Output/signals.json', 'r'))

    return render_template('dashboard.html', result_data=result_data, sentiments=sentiments,signals = signals)

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

@app.route('/TCS Prediction', methods=['GET', 'POST'])
def TCS_pred():
    
    file_path = [file for file in file_paths if file.endswith('TCS_data.csv')]
    ta = TechnicalAnalysis(file_path)
    
    fig1 = ta.ichimoku_graph(file_path[0])
    graph_div = fig1.to_html(full_html=False)
    return render_template('TCS_pred.html',graph_div=graph_div)
    

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

@app.route('/HDFC Prediction', methods=['GET', 'POST'])
def HDFC_pred():
    
    file_path = [file for file in file_paths if file.endswith('HDFCBANK_data.csv')]
    ta = TechnicalAnalysis(file_path)
    
    fig1 = ta.ichimoku_graph(file_path[0])
    graph_div = fig1.to_html(full_html=False)
    return render_template('HDFC_pred.html',graph_div=graph_div)

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

@app.route('/L&T Prediction', methods=['GET', 'POST'])
def LNT_pred():
    
    file_path = [file for file in file_paths if file.endswith('LT_data.csv')]
    ta = TechnicalAnalysis(file_path)
    
    fig1 = ta.ichimoku_graph(file_path[0])
    graph_div = fig1.to_html(full_html=False)
    return render_template('L&T_pred.html',graph_div=graph_div)

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

@app.route('/TITAN Prediction', methods=['GET', 'POST'])
def TITAN_pred():
    
    file_path = [file for file in file_paths if file.endswith('TITAN_data.csv')]
    ta = TechnicalAnalysis(file_path)
    
    fig1 = ta.ichimoku_graph(file_path[0])
    graph_div = fig1.to_html(full_html=False)
    return render_template('L&T_pred.html',graph_div=graph_div)


if __name__ == '__main__':
    app.run(host = "0.0.0.0",port = 8080)
    