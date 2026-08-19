import asyncio
import json
import re


from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from crawl4ai import AsyncWebCrawler
from suppport_functions import extract_middle_part_of_url

""" TODO: Fazer com que possa extrair os links sem ter de sair do browser, ou seja, 
extrair os links de uma pagina, esperar, eu interativamente vou pra outra página e depois extraio novamente. 
Perguntar se todos os links ja foram extraidos. Se sim, fechar o browser, e extrair o texto e metadados """

# --- Fase 1: Obtenção de Links (Interativa) ---

pesquisa_tribunais = { "TRE": "https://www.dgsi.pt/jtre.nsf/Pesquisa+Livre?OpenForm", # Tribunal da Relação de Évora
                      "TRL": "https://www.dgsi.pt/jtrl.nsf/Pesquisa+Livre?OpenForm", # Tribunal da Relação de Lisboa
                      "TRC": "https://www.dgsi.pt/jtrc.nsf/Pesquisa+Livre?OpenForm" , # Tribunal da Relação de Coimbra
                      "TRG": "https://www.dgsi.pt/jtrg.nsf/Pesquisa+Livre?OpenForm", # Tribunal da Relação de Guimarães
                      "TRP": "https://www.dgsi.pt/jtrp.nsf/Pesquisa+Livre?OpenForm"} # Tribunal da Relação do Porto

""" TODO: Depois em vez de pesquisa livre, vamos procurar por descritores específicos, como violência doméstica, divórcio, etc. Mas não ainda. Vamos focar nisso depois """

def get_dgsi_acordao_links_interactive(tribunal,url_tribunal: str):
    """
    Abre um browser para o utilizador fazer a pesquisa manualmente e depois extrai os links da página de resultados.
    """
    chrome_options = Options()
    # Não usar headless para que o utilizador possa interagir
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        
        print("🔍 Passo 1: A navegar para a página de pesquisa do DGSI...")
        driver.get(url_tribunal)
        
        print("\n✅ Controlo manual ativado.")
        print("   1. Faça a sua pesquisa no browser.")
        print("   2. Navegue até à página com a lista de resultados.")
        print("   3. Volte a este terminal e pressione Enter para continuar.")
        input()
        
        print("\n🔍 Passo 2: A extrair links da página de resultados atual...")
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        acordao_links = set()
        for key in pesquisa_tribunais.keys():
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                if key == "TRE" and 'jtre.nsf' in href:
                    mid_part_url = extract_middle_part_of_url(url_tribunal)
                elif key == "TRL" and 'jtrl.nsf' in href:
                    mid_part_url = extract_middle_part_of_url(url_tribunal)
                elif key == "TRC" and 'jtrc.nsf' in href:
                    mid_part_url = extract_middle_part_of_url(url_tribunal)
                elif key == "TRG" and 'jtrg.nsf' in href:
                    mid_part_url = extract_middle_part_of_url(url_tribunal)
                else:
                    mid_part_url = extract_middle_part_of_url(url_tribunal) # outputs tribunal do porto
                    
                    
                if 'OpenDocument' in href and mid_part_url in href:
                    if not href.startswith('http'):
                        href = 'https://www.dgsi.pt' + href
                    acordao_links.add(href)
        
        if not acordao_links:
            print("❌ Nenhum link de 'Acórdão' encontrado na página. A sair.")
            return []

        print(f"✅ Foram encontrados {len(acordao_links)} links únicos de acórdãos.")
        return list(acordao_links)
        
    except Exception as e:
        print(f"❌ Ocorreu um erro durante a extração de links: {e}")
        return []
    finally:
        print("🚪 A fechar o browser da fase de pesquisa.")
        driver.quit()

def clean_descritores(descritores_raw):
    """
    Limpa e valida a lista de descritores extraídos
    """
    if not descritores_raw:
        return []
    
    # Split por vírgula, ponto e vírgula ou quebra de linha
    descritores_raw_split = re.split(r'[;,\n]+', descritores_raw)
    
    # Lista de padrões a excluir (campos que não são descritores)
    exclude_patterns = [
        r'data\s+do\s+acord[aã]o',
        r'processo:',
        r'relator:',
        r'tribunal:',
        r'secção:',
        r'meio\s+processual:',
        r'decisão:',
        r'votação:',
        r'texto\s+integral',
        r'^[0-9\s\-/]+$'  # apenas números e traços (como datas ou números de processo)
    ]
    
    valid_descritores = []
    for d in descritores_raw_split:
        d_clean = d.strip()
        if len(d_clean) > 2:  # Mínimo de 3 caracteres
            # Verificar se não corresponde aos padrões de exclusão
            is_excluded = any(re.match(pattern, d_clean, re.IGNORECASE) for pattern in exclude_patterns)
            if not is_excluded:
                valid_descritores.append(d_clean)
    
    return valid_descritores

def extract_sumario_advanced(soup):
    """
    Extrai o sumário usando múltiplas estratégias para maior robustez.
    Retorna o sumário limpo ou None se não encontrado.
    """
    sumario_text = None
    
    # Estratégia 1: Buscar por tag <b> ou <strong> com "Sumário"
    sumario_tags = soup.find_all(['b', 'strong'], string=re.compile(r'sum[aá]rio', re.IGNORECASE))
    
    for tag in sumario_tags:
        content_parts = []
        
        # Buscar conteúdo nos siblings seguintes
        for sibling in tag.find_next_siblings():
            if not sibling:
                continue
                
            # Parar se encontrar outros títulos principais
            if sibling.name in ['b', 'strong'] and re.search(
                r'(decis[aã]o|acordam|relatório|voto|fundamentos)', 
                sibling.get_text(), 
                re.IGNORECASE
            ):
                break
                
            # Extrair texto do sibling
            text = sibling.get_text(strip=True) if hasattr(sibling, 'get_text') else str(sibling).strip()
            if text and len(text) > 10:  # Ignorar textos muito curtos
                content_parts.append(text)
        
        if content_parts:
            sumario_text = "\n".join(content_parts).strip()
            break
    
    # Estratégia 2: Buscar por regex no texto completo se Estratégia 1 falhou
    if not sumario_text:
        full_text = soup.get_text(separator='\n', strip=True)
        
        # Padrão específico: capturar tudo entre "Sumário:" e "Decisão Texto Integral/Parcial:"
        sumario_match = re.search(
            r'sum[aá]rio:?\s*\n?(.*?)(?=\n\s*(?:decis[aã]o\s+texto\s+(?:integral|parcial)|decis[aã]o|relatório|acordam|fundamentos|voto):|$)', 
            full_text, 
            re.IGNORECASE | re.DOTALL
        )
        
        if sumario_match:
            sumario_text = sumario_match.group(1).strip()
    
    # Estratégia 3: Buscar em estrutura de tabela (comum no DGSI)
    if not sumario_text:
        table_cells = soup.find_all(['td', 'th'])
        for cell in table_cells:
            cell_text = cell.get_text(strip=True)
            if re.match(r'sum[aá]rio:?', cell_text, re.IGNORECASE):
                # Buscar célula seguinte ou próximo sibling
                next_cell = cell.find_next_sibling(['td', 'th'])
                if next_cell:
                    sumario_text = next_cell.get_text(strip=True)
                    break
    
    # Limpeza final e validação
    if sumario_text:
        # Remover linhas muito curtas e limpeza geral
        lines = [line.strip() for line in sumario_text.split('\n') if line.strip()]
        clean_lines = [line for line in lines if len(line) > 5]  # Remover linhas muito curtas
        
        if clean_lines:
            sumario_text = "\n".join(clean_lines)
            
            # Validação: sumário deve ter pelo menos 50 caracteres para ser válido
            if len(sumario_text) >= 50:
                return sumario_text
    
    return None

# --- Fase 2: Scraping e Extração Estruturada ---

def extract_metadata_from_soup(soup, url, tribunal_nome=None, tipo_direito=None, tipo_caso=None):
    """
    Extrai todos os metadados estruturados do HTML de um acórdão.
    Suporta campos comuns a todos os tribunais + campos específicos.
    """
    metadata = {
        # Campos obrigatórios (identificação)
        "url": url,
        "tribunal": tribunal_nome,  # Controlado manualmente
        "tipo_direito": tipo_direito,  # Ex: "PENAL", "CÍVEL", "ADMINISTRATIVO"
        "tipo_caso": tipo_caso,  # Ex: "VIOLÊNCIA DOMÉSTICA", "DIVÓRCIO"
        
        # Campos comuns (extraídos automaticamente)
        "n_processo": None,
        "juiz_relator": None,
        "data_acordao": None,
        "descritores": [],  # Array de múltiplos descritores
        "votacao": None,
        "meio_processual": None,
        "decisao": None,
        "sumario": None,
        
        # Campos booleanos/específicos
        "texto_integral_disponivel": None,  # True/False baseado em "S/N"
        "texto_integral_completo": None,  # O texto HTML completo
        
        # Campos específicos por tribunal (flexíveis)
        "campos_especificos": {}
    }

    # Extração baseada em texto e padrões (regex)
    text = soup.get_text(separator='\n', strip=True)
    
    # Padrões Regex para campos comuns (todos os tribunais)
    patterns = {
        "n_processo": r"Processo:\s*([^\n]+)",
        "juiz_relator": r"Relator:\s*([^\n]+)",
        "data_acordao": r"Data do Acordão:\s*(\d{2}-\d{2}-\d{4})",
        "votacao": r"Votação:\s*([^\n]+)",
        "meio_processual": r"Meio Processual:\s*([^\n]+)",
        "decisao": r"Decisão:\s*([^\n]+)",
        "texto_integral_disponivel": r"Texto Integral:\s*([SN])",  # S ou N apenas
    }
    
    

    # Extrair campos básicos (EXCETO sumário - que tem função dedicada)
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key == "data_acordao":
                try:
                    metadata[key] = datetime.strptime(value, "%d-%m-%Y").strftime("%Y-%m-%d")
                except ValueError:
                    pass  # Deixa como None se o formato for inesperado
            elif key == "texto_integral_disponivel":
                metadata[key] = (value.upper() == "S")  # Boolean: True se "S", False se "N"
            else:
                metadata[key] = value

    # Extrair sumário usando função avançada dedicada
    metadata["sumario"] = extract_sumario_advanced(soup)

    # Extrair MÚLTIPLOS DESCRITORES (melhor parsing)
    descritores_match = re.search(r"Descritores:\s*([^\n]+(?:\n[^\n:]*)*)", text, re.IGNORECASE | re.MULTILINE)
    if descritores_match:
        descritores_text = descritores_match.group(1).strip()
        metadata["descritores"] = clean_descritores(descritores_text)
    else:
        metadata["descritores"] = []

    # Extrair Sumário usando função avançada
    metadata["sumario"] = extract_sumario_advanced(soup)

    # Extrair APENAS o texto integral (da "Decisão" para a frente) - poupa tokens
    for element in soup(["script", "style", "nav", "header", "footer", "form"]):
        element.decompose()
    
    full_text = soup.get_text(separator='\n', strip=True)
    
    # Procurar onde começa "Decisão Texto Integral:" 
    decisao_match = re.search(
        r'(decis[aã]o\s+texto\s+integral:.*)', 
        full_text, 
        re.IGNORECASE | re.DOTALL
    )
    
    if decisao_match:
        # Capturar apenas da decisão para a frente (economiza tokens)
        metadata["texto_integral_completo"] = decisao_match.group(1).strip()
    else:
        # Fallback: se não encontrar a marcação, usar texto completo
        metadata["texto_integral_completo"] = full_text

    return metadata

async def scrape_and_structure_acordao(url, session_id, total, tribunal_config=None):
    """
    Faz o scrape de um acórdão, extrai os dados de forma estruturada e retorna um dicionário.
    Execução sequencial - sem concorrência.
    
    tribunal_config: Dict com 'nome', 'tipo_direito', 'tipo_caso' para controlo manual
    """
    print(f"[{session_id}/{total}] 🟢 A iniciar scrape para: {url[:80]}...")
    
    # Configuração padrão se não fornecida
    if tribunal_config is None:
        tribunal_config = {
            "nome": "TRP",  # Tribunal da Relação do Porto (padrão)
            "tipo_direito": "PENAL", 
            "tipo_caso": "VIOLÊNCIA DOMÉSTICA"
        }
    
    try:
        async with AsyncWebCrawler(headless=True) as crawler:
            result = await crawler.arun(url=url)
        
        if result.status_code == 200:
            soup = BeautifulSoup(result.html, 'html.parser')
            structured_data = extract_metadata_from_soup(
                soup, 
                url, 
                tribunal_nome=tribunal_config["nome"],
                tipo_direito=tribunal_config["tipo_direito"],
                tipo_caso=tribunal_config["tipo_caso"]
            )
            print(f"[{session_id}/{total}] ✅ Sucesso: {url[:80]}")
            return {"status": "success", "data": structured_data}
        else:
            print(f"[{session_id}/{total}] ⚠️ Erro HTTP {result.status_code}: {url[:80]}")
            return {"status": "error", "url": url, "details": f"HTTP status {result.status_code}"}
    except Exception as e:
        error_details = repr(e) # Usar repr() para obter mais detalhes do erro
        print(f"[{session_id}/{total}] ❌ Exceção: {url[:80]} - {error_details}")
        return {"status": "error", "url": url, "details": error_details}
    
""" TODO: Separar as funções de extração da função de orquestração """

# --- Fase 3: Orquestração ---

async def main():
    # Perguntar ao utilizador qual o tribunal a pesquisar
    print("🏛️ Escolha o tribunal para a pesquisa:")
    response = input(f"Escolha entre {','.join(pesquisa_tribunais.keys())}:").strip().upper()
    
    if response not in pesquisa_tribunais:
        print("❌ Tribunal inválido. A sair.")
        return  
    else:
        url_tribunal = pesquisa_tribunais[response]
    # Fase 1
    acordao_links = get_dgsi_acordao_links_interactive(response,url_tribunal=url_tribunal)
    
    if not acordao_links:
        return

    # Fase 2 - Seleção de quantidade e Scraping
    total_links_found = len(acordao_links)
    
    # Perguntar ao utilizador quantos links quer processar
    try:
        response = input(f"\nForam encontrados {total_links_found} links. Quantos deseja processar? (e.g., 5, 10, ou 'all'): ")
        if response.lower() == 'all':
            num_to_scrape = total_links_found
        else:
            num_to_scrape = int(response)
            if num_to_scrape > total_links_found:
                print(f"Aviso: O número pedido ({num_to_scrape}) é maior que o total de links encontrados ({total_links_found}). A processar todos.")
                num_to_scrape = total_links_found
    except (ValueError, TypeError):
        print("Entrada inválida. A processar os primeiros 5 links por defeito.")
        num_to_scrape = 5

    links_to_scrape = acordao_links[:num_to_scrape]
    
    # Configuração manual do tribunal (flexível)
    """ TODO: Melhorar esta parte para evitar input manual repetido """
    
    
    print(f"\n🏛️ Configuração do Tribunal :")
    tribunal_nome = input("Nome do tribunal (choose between: TRP, TRL, TRC, TRG, TRE) [padrão: TRP]: ").strip() or "TRP"
    tipo_direito = input("Tipo de direito (choose between: PENAL, CÍVEL, TRABALHO ) [padrão: PENAL]: ").strip() or "PENAL"
    tipo_caso = input("Tipo de caso (ex: VIOLÊNCIA DOMÉSTICA) [padrão: VIOLÊNCIA DOMÉSTICA]: ").strip() or "VIOLÊNCIA DOMÉSTICA"
    
    tribunal_config = {
        "nome": tribunal_nome,
        "tipo_direito": tipo_direito,
        "tipo_caso": tipo_caso
    }
    
    print(f"\nFase 2: A fazer o scrape de {len(links_to_scrape)} acórdãos sequencialmente (um de cada vez)...")
    
    # Execução SEQUENCIAL - processar um URL de cada vez
    results = []
    for i, link in enumerate(links_to_scrape):
        result = await scrape_and_structure_acordao(link, i+1, len(links_to_scrape), tribunal_config)
        results.append(result)
        
        # Pequena pausa entre requests para ser gentil com o servidor
        await asyncio.sleep(1)
    
    # Fase 3
    successful_scrapes = [res['data'] for res in results if res and res['status'] == 'success']
    failed_scrapes = [res for res in results if res and res['status'] == 'error']
    
    print(f"\n✅ Scraping completo. Sucessos: {len(successful_scrapes)}, Falhas: {len(failed_scrapes)}")
    
    if successful_scrapes:
        output_filename = "dgsi_acordaos_structured_v5.json"  # Nome atualizado
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(successful_scrapes, f, ensure_ascii=False, indent=4)
        print(f"💾 Todos os dados estruturados foram guardados em {output_filename}")

    if failed_scrapes:
        with open("failed_urls.txt", "w", encoding='utf-8') as f:
            for failure in failed_scrapes:
                f.write(f"{failure['url']} - Motivo: {failure['details']}\n")
        print("⚠️ Alguns URLs falharam. Verifique failed_urls.txt para mais detalhes.")

if __name__ == "__main__":
    asyncio.run(main())
