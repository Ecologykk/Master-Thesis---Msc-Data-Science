import re
import json
from pathlib import Path

def limpar_texto_integral(csm_json):
    """
    Remove tudo até e incluindo 'Decisão Texto Integral' e retorna apenas o conteúdo que vem depois.
    Salva o texto limpo no próprio JSON, substituindo o campo 'texto_integral_completo'.
    
    Args:
        csm_json (dict): objeto JSON do csm contendo 'texto_integral_completo'
        
    Returns:
        dict: O mesmo JSON, com 'texto_integral_completo' corrigido
    """
    texto = csm_json.get('texto_integral_completo')
    

    if texto is None or texto == "":
        return csm_json
    
    

    text = str(texto)
    pattern = r'(?:decisão|decisao)\s+texto\s+integral(?:\s*[:\-–—]\s*)?(.*)'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

    if match:
        texto_limpo = match.group(1).lstrip()
    else:
        texto_limpo = text

    csm_json['texto_integral_completo'] = texto_limpo
    return csm_json


def extrair_data_acordao(texto):
    """
    Extrai a data do acórdão do texto fornecido.
    
    Args:
        texto (str): texto do acórdão
        
    Returns:
        str ou None: data no formato 'dd/mm/aaaa' ou None se não encontrada
    """
    pattern = r'Data do Acord[ãa]o[:\s]*(\d{2}/\d{2}/\d{4})'
    match = re.search(pattern, texto, re.IGNORECASE)
    
    if match:
        return match.group(1)
    return None


def limpar_texto_integral_sem_decisao(csm_json):
    """
    Remove tudo até e incluindo 'Decisão Texto Integral' e retorna apenas o conteúdo que vem depois.
    Salva o texto limpo no próprio JSON, substituindo o campo 'texto_integral_sem_decisao'.
    
    Args:
        csm_json (dict): objeto JSON do csm contendo 'texto_integral_sem_decisao'
        
    Returns:
        dict: O mesmo JSON, com 'texto_integral_sem_decisao' corrigido
        
    """

    texto = csm_json.get('texto_integral_sem_decisao')
    
    data_acordao = extrair_data_acordao(texto)
    
    if data_acordao:
            csm_json['data_acordao'] = data_acordao
        
            

    if texto is None or texto == "":
        return csm_json
    
    

    text = str(texto)
    pattern = r'(?:decisão|decisao)\s+texto\s+integral(?:\s*[:\-–—]\s*)?(.*)'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

    if match:
        texto_limpo = match.group(1).lstrip()
    else:
        texto_limpo = text

    csm_json['texto_integral_sem_decisao'] = texto_limpo
    return csm_json




def salvar_json(csm_json, arquivo_path):
    """
    Salva o JSON corrigido no arquivo.
    
    Args:
        csm_json (dict): objeto JSON corrigido
        arquivo_path (str): caminho do arquivo JSON
    """
    with open(arquivo_path, 'w', encoding='utf-8') as f:
        json.dump(csm_json, f, ensure_ascii=False, indent=2)
        
        
        
def main():
    
    json_file_path = Path('./../../data/csm_violencia_domestica_20251029_131345.json')
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    casos_processados = 0
    
    for item in data:
        # texto_original = item.get('texto_integral_completo', '')
        texto_sem_decisao = item.get('texto_integral_sem_decisao', '')
        texto_original = texto_sem_decisao
        if texto_original:
            # limpar_texto_integral(item)  
            limpar_texto_integral_sem_decisao(item)
            casos_processados += 1
            
    salvar_json(data, json_file_path)
    print(f"✅ Ficheiro atualizado: {json_file_path}")
    print(f"   📊 {casos_processados} casos processados e texto limpo")
    
    
    
if __name__ == "__main__":
    main()
    
            

# \nData do Acordão:\n28/05/2002