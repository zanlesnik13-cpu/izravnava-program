import sys
from pathlib import Path

# Zagotovitev, da je korenski imenik aplikacije vedno v sys.path (za Streamlit Cloud in lokalni zagon)
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
