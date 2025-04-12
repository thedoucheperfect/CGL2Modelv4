# app.py (updated)
import os
from flask import Flask, render_template, request, session
import pandas as pd
from ModelUser import ModelHandler
from weather import get_wbt  # New import

app = Flask(__name__)
app.secret_key = "8fF5R^:RwP.t1D%vMH4N^12*7&=+"
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 600
app.config['SESSION_TYPE'] = 'filesystem'

@app.route('/', methods=['GET', 'POST'])
def index():
    grade_file = os.path.join('Model Files', 'GradeFile', 'GradeFile.xlsx')
    df = pd.read_excel(grade_file)
    grades = df['RM Grade'].tolist()
    
    session.modified = True
    
    if request.method == 'POST':
        # Store user inputs
        session['inputs'] = {
            'width': float(request.form['width']),
            'thickness': float(request.form['thickness']),
            'gsm_a': float(request.form['gsm_a']),
            'hardness': float(request.form['hardness']),
            'JCFEX_STRIP': float(request.form['dripping_temp']),
            'rm_grade': request.form['rm_grade']
        }
        
        # Get predictions
        handler = ModelHandler()
        main_outputs, firing = handler.predict_all(session['inputs'])
        
        # Fetch weather data and calculate WBT+4
        wbt = get_wbt()
        wbt_plus_4 = round(wbt + 4) if wbt is not None else None
        jcf = main_outputs['JCF']
        max_jcwt = round(max(jcf, wbt_plus_4)) if wbt_plus_4 else round(jcf)

        # Prepare outputs
        rounded_outputs = {k: round(v) for k, v in main_outputs.items()}
        tph = round((60 * 7850 * session['inputs']['width'] * 
                   session['inputs']['thickness'] * main_outputs['Speed']) / 1e9)

        return render_template('Output.html',
                             inputs=session['inputs'],
                             outputs=rounded_outputs,
                             firing=round(firing),
                             tph=tph,
                             wbt_plus_4=wbt_plus_4,
                             max_jcwt=max_jcwt)

    return render_template('InputForm.html', 
                         grades=grades,
                         inputs=session.get('inputs', {}))

if __name__ == '__main__':
    app.run(debug=True)
