import re

def clean_code(code: str) -> str:
    # Rimuove linee vuote multiple
    code = re.sub(r'\n\s*\n', '\n\n', code)
    # Spazi finali
    code = re.sub(r'[ \t]+$', '', code, flags=re.MULTILINE)
    # Puoi aggiungere qui rimozione di blocchi troppo piccoli ecc.
    return code
