import os
import json
import numpy as np
import pandas as pd
from joblib import load

class ModelHandler:
    def __init__(self):
        # Load models with validation
        self.main_models = {
            'rfr': load(os.path.join('Model Files', 'rfr_main.joblib')),
            'xgb': load(os.path.join('Model Files', 'xgb_main.joblib'))
        }
        
        self.firing_models = {
            'rfr': load(os.path.join('Model Files', 'rfr_firing.joblib')),
            'xgb': load(os.path.join('Model Files', 'xgb_firing.joblib'))
        }
        
        # Validate scalers
        self.scaler = self._validate_scaler(
            load(os.path.join('Model Files', 'scaler.joblib')),
            expected_features=21
        )
        self.scaler_firing = self._validate_scaler(
            load(os.path.join('Model Files', 'scaler_firing.joblib')),
            expected_features=6
        )
        
        # Load composition data
        self.grade_df = pd.read_excel(
            os.path.join('Model Files', 'GradeFile', 'GradeFile.xlsx')
        )
        
        # Load R² scores
        with open(os.path.join('Model Files', 'json files', 'main_r2_scores.json')) as f:
            self.main_r2 = json.load(f)['R2_scores']
            
        with open(os.path.join('Model Files', 'json files', 'firing_r2_scores.json')) as f:
            self.firing_r2 = json.load(f)['R2_scores_Firing']

    def _validate_scaler(self, scaler, expected_features):
        if scaler.n_features_in_ != expected_features:
            raise ValueError(f"Scaler expects {scaler.n_features_in_} features, "
                             f"but {expected_features} required")
        return scaler

    def _get_composition(self, rm_grade):
        row = self.grade_df[self.grade_df['RM Grade'] == rm_grade].iloc[0]
        return row[['C','Mn','P','S','Si','Al','N','MAE','Ni','Cu','Mo',
                   'B','Ti','Nb','V','Cr']].values

    def predict_main(self, inputs):
        # Manual features FIRST (matches training order)
        manual_features = [
            inputs['width'],        # Width
            inputs['thickness'],    # Thickness
            inputs['gsm_a'],        # GSM-A
            inputs['hardness'],     # Hardness
            inputs['JCFEX_STRIP']   # Dipping Temperature
        ]
        
        # Add composition elements
        composition = self._get_composition(inputs['rm_grade'])
        X = manual_features + composition.tolist()
        
        # Validate feature count
        if len(X) != 21:
            raise ValueError(f"Expected 21 features, got {len(X)}")
        
        X_scaled = self.scaler.transform([X])
        
        # Get predictions from both models
        rf_pred = self.main_models['rfr'].predict(X_scaled)[0]  # Shape: (n_outputs,)
        xgb_pred = self.main_models['xgb'].predict(X_scaled)[0]  # Shape: (n_outputs,)
        
        outputs = {}
        for idx, target in enumerate(self.main_r2.keys()):
            if self.main_r2[target]['Random Forest'] > self.main_r2[target]['XGBoost']:
                outputs[target] = rf_pred[idx].item()  # Convert numpy type to Python float
            else:
                outputs[target] = xgb_pred[idx].item()
            
        return outputs

    def predict_firing(self, nof_values, speed):
        # Convert to numpy array and ensure 2D shape
        X = np.array(nof_values + [speed]).reshape(1, -1)
        
        if X.shape[1] != 6:
            raise ValueError(f"Expected 6 features, got {X.shape[1]}")
        
        X_scaled = self.scaler_firing.transform(X)
        
        if self.firing_r2['Random Forest'] > self.firing_r2['XGBoost']:
            model = self.firing_models['rfr']
        else:
            model = self.firing_models['xgb']
            
        return model.predict(X_scaled)[0].item()  # Return Python float

    def predict_all(self, inputs):
        main_outputs = self.predict_main(inputs)
        
        # Prepare firing model inputs
        nof_values = [main_outputs[f'NOF{i}'] for i in range(1,6)]
        firing = self.predict_firing(nof_values, main_outputs['Speed'])
        
        # Remove NOF1-5 from final outputs
        for i in range(1,6):
            del main_outputs[f'NOF{i}']
            
        return main_outputs, firing
