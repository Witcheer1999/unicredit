
import json

notebook_path = '/Users/michael/Desktop/unicredit/churn prediction ita.ipynb'

# Markdown texts
md_intro = """# 8. Ottimizzazione Bayesiana e Approfondimento Modelli

In questa sezione avanzata, approfondiremo l'analisi utilizzando l'**Ottimizzazione Bayesiana** per la ricerca degli iperparametri ottimali.
Prima di procedere, analizziamo le peculiarità logiche dei quattro modelli che utilizzeremo.

## 8.1 Peculiarità Logiche dei Modelli

### 1. Regressione Logistica (Logistic Regression)
La Regressione Logistica è un modello **lineare** utilizzato per la classificazione binaria.
*   **Logica**: Stima la probabilità che un'istanza appartenga a una classe applicando la funzione sigmoide a una combinazione lineare delle feature.
*   **Caratteristiche**:
    *   **Interpretabilità**: I coefficienti indicano direttamente l'impatto di ogni feature (al netto di trasformazioni non lineari).
    *   **Linearità**: Assume una relazione lineare tra le variabili indipendenti e il log-odds della variabile dipendente.
    *   **Probabilità**: Fornisce nativamente delle probabilità ben calibrate.

### 2. Support Vector Machine (SVM)
Le SVM cercano l'iperpiano che separa le classi massimizzando il **margine** (la distanza tra l'iperpiano e i punti più vicini di ciascuna classe, detti vettori di supporto).
*   **Logica**: I dati vengono mappati in uno spazio dimensionale superiore (tramite il "kernel trick") dove è più probabile che siano linearmente separabili.
*   **Caratteristiche**:
    *   **Kernel Trick**: Permette di modellare confini decisionali non lineari complessi senza dover calcolare esplicitamente le coordinate nello spazio ad alta dimensione.
    *   **Robustezza**: Efficace in spazi ad alta dimensione.
    *   **Sensibilità**: Richiede feature scalate (es. StandardScaler) ed è sensibile al rumore vicino al confine di decisione.

### 3. Random Forest
Random Forest è un metodo **ensemble** basato sul **Bagging** (Bootstrap Aggregating) di alberi decisionali.
*   **Logica**: Costruisce numerosi alberi decisionali su sottoinsiemi casuali dei dati e delle feature, e aggrega le loro predizioni (moda per classificazione) per ridurre la varianza.
*   **Caratteristiche**:
    *   **Robustezza**: Riduce il rischio di overfitting rispetto ai singoli alberi decisionali.
    *   **Non Linearità**: Cattura relazioni non lineari complesse.
    *   **Feature Importance**: Fornisce una stima dell'importanza relativa di ogni variabile.

### 4. XGBoost (Extreme Gradient Boosting)
XGBoost è un algoritmo di **ensemble** basato sul **Gradient Boosting**.
*   **Logica**: Costruisce il modello in modo sequenziale. Ogni nuovo albero cerca di correggere gli errori (residui) commessi dagli alberi precedenti, minimizzando una funzione di perdita regolarizzata.
*   **Caratteristiche**:
    *   **Boosting**: Focalizza l'attenzione sui casi "difficili" da classificare.
    *   **Regolarizzazione**: Include termini di penalità per controllare la complessità del modello e prevenire l'overfitting.
    *   **Efficienza**: Ottimizzato per velocità e prestazioni, gestisce nativamente i valori mancanti.
"""

# Code cells
code_install = "!pip install scikit-optimize xgboost"

code_imports = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import warnings
warnings.filterwarnings('ignore')
"""

code_prep = """
# Ricarichiamo/Prepariamo i dati per sicurezza partendo da 'df' (assumendo che sia il dataframe pulito definito nelle celle precedenti)
# Se 'df' non dovesse essere disponibile, decommentare le righe sotto per ricaricarlo
# try:
#     df
# except NameError:
#     df = pd.read_csv('Churn_Modelling.csv')
#     df = df.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)

# Encoding
le = LabelEncoder()
df_opt = df.copy()
if df_opt['Gender'].dtype == 'object':
    df_opt['Gender'] = le.fit_transform(df_opt['Gender'])
if df_opt['Geography'].dtype == 'object':
    df_opt = pd.get_dummies(df_opt, columns=['Geography'], drop_first=True)

# Split
X = df_opt.drop('Exited', axis=1)
y = df_opt['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling (Importante per SVM e LR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Shape:", X_train.shape, X_test.shape)
"""

code_lr = """
# 8.2 Ottimizzazione Bayesiana: Logistic Regression

print("Inizio ottimizzazione Logistic Regression...")
opt_lr = BayesSearchCV(
    LogisticRegression(solver='liblinear'),
    {
        'C': Real(1e-2, 1e+2, prior='log-uniform'),
        'penalty': Categorical(['l1', 'l2'])
    },
    n_iter=20,
    cv=5,
    random_state=42,
    n_jobs=-1
)

opt_lr.fit(X_train_scaled, y_train)

print("Best params (LR):", opt_lr.best_params_)
print("Best score (LR):", opt_lr.best_score_)
y_pred_lr = opt_lr.predict(X_test_scaled)
print("Risultati Logistic Regression:")
print(classification_report(y_test, y_pred_lr))
"""

code_svm = """
# 8.3 Ottimizzazione Bayesiana: Support Vector Machine (SVM)

print("Inizio ottimizzazione SVM...")
opt_svm = BayesSearchCV(
    SVC(probability=True),
    {
        'C': Real(1e-1, 1e+2, prior='log-uniform'),
        'gamma': Real(1e-3, 1e-1, prior='log-uniform'),
        'kernel': Categorical(['linear', 'rbf'])
    },
    n_iter=20,
    cv=5,
    random_state=42,
    n_jobs=-1
)

opt_svm.fit(X_train_scaled, y_train)

print("Best params (SVM):", opt_svm.best_params_)
print("Best score (SVM):", opt_svm.best_score_)
y_pred_svm = opt_svm.predict(X_test_scaled)
print("Risultati SVM:")
print(classification_report(y_test, y_pred_svm))
"""

code_rf = """
# 8.4 Ottimizzazione Bayesiana: Random Forest

print("Inizio ottimizzazione Random Forest...")
opt_rf = BayesSearchCV(
    RandomForestClassifier(random_state=42),
    {
        'n_estimators': Integer(50, 300),
        'max_depth': Integer(3, 20),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 10)
    },
    n_iter=20,
    cv=5,
    random_state=42,
    n_jobs=-1
)

opt_rf.fit(X_train, y_train)

print("Best params (RF):", opt_rf.best_params_)
print("Best score (RF):", opt_rf.best_score_)
y_pred_rf = opt_rf.predict(X_test)
print("Risultati Random Forest:")
print(classification_report(y_test, y_pred_rf))
"""

code_xgb = """
# 8.5 Ottimizzazione Bayesiana: XGBoost

print("Inizio ottimizzazione XGBoost...")
opt_xgb = BayesSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    {
        'learning_rate': Real(0.01, 1.0, prior='log-uniform'),
        'n_estimators': Integer(50, 300),
        'max_depth': Integer(3, 10),
        'subsample': Real(0.5, 1.0),
        'colsample_bytree': Real(0.5, 1.0)
    },
    n_iter=20,
    cv=5,
    random_state=42,
    n_jobs=-1
)

opt_xgb.fit(X_train, y_train)

print("Best params (XGB):", opt_xgb.best_params_)
print("Best score (XGB):", opt_xgb.best_score_)
y_pred_xgb = opt_xgb.predict(X_test)
print("Risultati XGBoost:")
print(classification_report(y_test, y_pred_xgb))
"""

def create_cell(source, cell_type='code'):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "outputs": [],
        "source": source.split('\\n') # split into lines for notebook format? 
                                     # actually notebook format expects a list of strings, 
                                     # but mostly usually simple splitting is fine or just keeping it as list.
                                     # Let's keep it simple: list of strings with \n.
    }
    
# Better helper to split correctly ensuring newlines
def to_source_list(text):
    lines = text.splitlines(keepends=True)
    # If the last line doesn't have a newline, it's fine.
    # We might need to handle empty initial newline from triple quotes
    if lines and lines[0].strip() == '':
        lines = lines[1:]
    return lines

new_cells_content = [
    (md_intro, 'markdown'),
    (code_install, 'code'),
    (code_imports, 'code'),
    (code_prep, 'code'),
    (code_lr, 'code'),
    (code_svm, 'code'),
    (code_rf, 'code'),
    (code_xgb, 'code')
]

# Read
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Create new cells objects
for content, ctype in new_cells_content:
    new_cell = {
        "cell_type": ctype,
        "metadata": {},
        "source": to_source_list(content)
    }
    if ctype == 'code':
        new_cell["outputs"] = []
        new_cell["execution_count"] = None
        
    nb['cells'].append(new_cell)

# Write
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("Notebook updated successfully.")
