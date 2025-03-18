import os
from flask import Flask, render_template, request, session
import pandas as pd
from ModelUser import ModelHandler

app = Flask(__name__)
app.secret_key = "8fF5R^:RwP.t1D%vMH4N^12*7&=+"  # Fixed key
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 600  # 10 minutes
app.config['SESSION_TYPE'] = 'filesystem'

@app.route('/', methods=['GET', 'POST'])
def index():
    grade_file = os.path.join('Model Files', 'GradeFile', 'GradeFile.xlsx')
    df = pd.read_excel(grade_file)
    grades = df['RM Grade'].tolist()
    
    # Preserve session across requests
    session.modified = True
    
    if request.method == 'POST':
        session['inputs'] = {
            'width': float(request.form['width']),
            'thickness': float(request.form['thickness']),
            'gsm_a': float(request.form['gsm_a']),
            'hardness': float(request.form['hardness']),
            'JCFEX_STRIP': float(request.form['dripping_temp']),
            'rm_grade': request.form['rm_grade']
        }
        
        handler = ModelHandler()
        main_outputs, firing = handler.predict_all(session['inputs'])
        
        rounded_outputs = {k: round(v) for k, v in main_outputs.items()}
        tph = round((60 * 7850 * session['inputs']['width'] * 
                   session['inputs']['thickness'] * main_outputs['Speed']) / 1e9)
        
        return render_template('Output.html',
                             inputs=session['inputs'],
                             outputs=rounded_outputs,
                             firing=round(firing),
                             tph=tph)
    
    # Explicitly load saved inputs for GET requests
    return render_template('InputForm.html', 
                         grades=grades,
                         inputs=session.get('inputs', {}))

if __name__ == '__main__':
    app.run(debug=True)
