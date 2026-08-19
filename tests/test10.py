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




# --- Fase 1: Obtenção de Links (Interativa) ---

# URLs de pesquisa por descritores - Tribunais de Relação (Violência Doméstica)
pesquisa_descritores_tr = { 
    "TRE": "https://www.dgsi.pt/jtre.nsf/Pesquisa+Descritor?OpenForm", # Tribunal da Relação de Évora
    "TRL": "https://www.dgsi.pt/jtrl.nsf/Pesquisa+Descritor?OpenForm", # Tribunal da Relação de Lisboa
    "TRC": "https://www.dgsi.pt/jtrc.nsf/Pesquisa+Descritor?OpenForm" , # Tribunal da Relação de Coimbra
    "TRG": "https://www.dgsi.pt/jtrg.nsf/Pesquisa+Descritor?OpenForm", # Tribunal da Relação de Guimarães
    "TRP": "https://www.dgsi.pt/jtrp.nsf/Pesquisa+Descritor?OpenForm" # Tribunal da Relação do Porto
}

# URL de pesquisa por campo - Julgados de Paz (Incumprimento de Contratos)
pesquisa_julgados_paz = "https://www.dgsi.pt/cajp.nsf/Pesquisa+Campo?OpenForm"

def get_dgsi_acordao_links_multipagina_interactive(tipo_pesquisa, url_tribunal, tribunal_id=None):
    """
    VERSÃO UNIFICADA: Mantém o browser aberto para múltiplas extrações de links.
    Suporta tanto Tribunais de Relação como Julgados de Paz.
    
    Args:
        tipo_pesquisa: "TR" (Tribunais Relação) ou "JP" (Julgados Paz)
        url_tribunal: URL da página de pesquisa
        tribunal_id: Para TR, o identificador (ex: "jtrl"). Para JP, será "cajp"
    """
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)
    all_acordao_links = set()  # Conjunto para evitar duplicados automaticamente
    
    try:
        print("🔍 Passo 1: A navegar para a página de pesquisa do DGSI...")
        driver.get(url_tribunal)
        
        print("\n✅ Controlo manual ativado - MODO MULTI-PÁGINA")
        print("   🎯 Este modo permite extrair links de múltiplas páginas sem fechar o browser.")
        print("\n" + "="*80)
        
        if tipo_pesquisa == "TR":
            print("⚠️  IMPORTANTE - FLUXO PARA TRIBUNAIS DE RELAÇÃO (VIOLÊNCIA DOMÉSTICA):")
            print("="*80)
            print("1. Pesquise por palavras-chave (ex: 'violência doméstica')")
            print("2. Verá uma lista de DESCRITORES possíveis")
            print("3. ⚡ CLIQUE num descritor específico (ex: 'VÍTIMA DE VIOLÊNCIA DOMÉSTICA')")
        else:  # JP
            print("⚠️  IMPORTANTE - FLUXO PARA JULGADOS DE PAZ (INCUMPRIMENTO DE CONTRATOS):")
            print("="*80)
            print("1. Pesquise por descritores/campos (ex: 'incumprimento de contrato')")
            print("2. Verá uma página com resultados")
        
        print("4. Agora verá a LISTA DE DECISÕES/ACÓRDÃOS")
        print("5. NESTA página de lista, volte aqui e pressione '1' para extrair")
        print("6. Se houver múltiplas páginas de resultados, navegue entre elas e repita")
        print("="*80 + "\n")
        
        while True:
            print(f"\n📊 Links acumulados até agora: {len(all_acordao_links)}")
            print("\n🔄 Opções disponíveis:")
            print("   1. Extrair links da página atual")
            print("   2. Terminar e continuar com o scraping")
            print("   3. Sair sem fazer scraping")
            
            escolha = input("\nEscolha uma opção (1-3): ").strip()
            
            if escolha == "1":
                print("\n🔍 A extrair links da página atual...")
                print(f"🔍 DEBUG: Tipo de pesquisa = {tipo_pesquisa}")
                print(f"🔍 DEBUG: URL do tribunal = {url_tribunal}")
                print(f"🔍 DEBUG: Tribunal ID = {tribunal_id}")
                
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                
                # DEBUG: Verificar URL atual no browser
                current_url = driver.current_url
                print(f"🔍 DEBUG: URL atual do browser = {current_url}")
                
                links_pagina_atual = set()
                all_links_found = []  # Para debug
                
                print(f"🔍 DEBUG: Procurando por links com '{tribunal_id}.nsf'")
                
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    all_links_found.append(href)  # Guardar todos para debug
                    
                    # Verificar se o link pertence ao tribunal correto
                    if f'{tribunal_id}.nsf' in href.lower():
                        # Aceitar links que contenham OpenDocument OU que pareçam IDs de documentos
                        is_valid_link = False
                        
                        # Método 1: Links diretos com OpenDocument
                        if 'OpenDocument' in href:
                            is_valid_link = True
                        
                        # Método 2: Links com estrutura de ID (hexadecimal de 32 caracteres)
                        # Exemplo: /jtrl.nsf/33182fc732316039802568d9003626b5/abc123...
                        # Exemplo JP: /cajp.nsf/38f33c44b0c8358280256879006bc013/abc123...
                        elif re.search(r'/[a-f0-9]{32}/', href, re.IGNORECASE):
                            is_valid_link = True
                        
                        # Método 3: Links simples com ID depois do .nsf/
                        # Exemplo: /jtrl.nsf/abc123def456 ou /cajp.nsf/abc123def456
                        elif re.match(rf'.*{tribunal_id}\.nsf/[a-f0-9]{{32}}', href, re.IGNORECASE):
                            is_valid_link = True
                        
                        # EXCLUIR: Links de pesquisa/navegação
                        if any(x in href for x in ['SearchView', 'CreateDocument', 'OpenForm', 'Pesquisa', 'Por+Ano', 'Pesquisa+Campo']):
                            is_valid_link = False
                        
                        if is_valid_link:
                            # Converter para URL absoluto se necessário
                            if not href.startswith('http'):
                                href = 'https://www.dgsi.pt' + href
                            links_pagina_atual.add(href)
                
                # DEBUG: Mostrar estatísticas
                print(f"\n📊 DEBUG: Total de links na página: {len(all_links_found)}")
                print(f"📊 DEBUG: Links do tribunal {tribunal_id}: {sum(1 for h in all_links_found if f'{tribunal_id}.nsf' in h.lower())}")
                print(f"📊 DEBUG: Links com 'OpenDocument': {sum(1 for h in all_links_found if 'OpenDocument' in h)}")
                print(f"📊 DEBUG: Links com IDs hexadecimais: {sum(1 for h in all_links_found if re.search(r'[a-f0-9]{32}', h, re.IGNORECASE))}")
                
                # Mostrar alguns exemplos de links encontrados do tribunal
                tribunal_links_examples = [h for h in all_links_found if f'{tribunal_id}.nsf' in h.lower()][:10]
                print(f"\n🔍 DEBUG: Primeiros {min(5, len(tribunal_links_examples))} links do {tribunal_id} encontrados:")
                for i, href in enumerate(tribunal_links_examples):
                    truncated = href[:120] + "..." if len(href) > 120 else href
                    print(f"  {i+1}. {truncated}")
                
                # Adicionar novos links ao conjunto total
                links_novos = links_pagina_atual - all_acordao_links
                all_acordao_links.update(links_pagina_atual)
                
                print(f"\n✅ Extraídos {len(links_pagina_atual)} links desta página")
                print(f"📈 Novos links únicos: {len(links_novos)}")
                print(f"📊 Total acumulado: {len(all_acordao_links)} links únicos")
                
                if len(links_pagina_atual) == 0:
                    print("\n⚠️  Nenhum link de acórdão/sentença encontrado nesta página.")
                    print("💡 POSSÍVEIS CAUSAS:")
                    print("   1. Está na página de LISTA DE DESCRITORES (precisa clicar num descritor)")
                    print("   2. Está na página de pesquisa (precisa fazer a pesquisa primeiro)")
                    print("   3. A página ainda está a carregar (aguarde e tente novamente)")
                    print("   4. Formato de links diferente (veja os exemplos acima)")
                else:
                    print(f"\n✅ Links extraídos com sucesso!")
                    if len(links_novos) > 0:
                        print(f"📝 Exemplos dos novos links encontrados:")
                        for i, link in enumerate(list(links_novos)[:3]):
                            print(f"  {i+1}. {link[:100]}...")
                
                print("\n👆 Navegue para a próxima página no browser e volte aqui para extrair mais links.")
                
            elif escolha == "2":
                if len(all_acordao_links) > 0:
                    print(f"\n✅ Finalizando extração com {len(all_acordao_links)} links únicos.")
                    return list(all_acordao_links)
                else:
                    print("❌ Nenhum link foi extraído. A sair.")
                    return []
                    
            elif escolha == "3":
                print("🚪 A sair sem fazer scraping.")
                return []
            else:
                print("❌ Opção inválida. Escolha 1, 2 ou 3.")
        
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
        r'data\s+da\s+senten[cç]a',  # NOVO: excluir "Data da sentença"
        r'processo:',
        r'relator:',
        r'tribunal:',
        r'secção:',
        r'meio\s+processual:',
        r'decisão:',
        r'votação:',
        r'texto\s+integral',
        r'julgado\s+de\s+paz',  # NOVO: excluir referências ao julgado
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

def extract_metadata_julgados_paz(soup, url):
    """
    Extrai metadados específicos dos Julgados de Paz.
    Campos: url, data, Relator, Descritores, Data da sentença, Julgado de Paz de, Decisão texto integral
    """
    metadata = {
        "url": url,
        "relator": None,
        "descritores": [],
        "data_sentenca": None,
        "julgado_paz_de": None,  # Ex: "Lisboa", "Porto", etc.
        "decisao_texto_integral": None
    }

    # Extração baseada em texto e padrões (regex)
    text = soup.get_text(separator='\n', strip=True)
    
    # Padrões Regex para campos específicos dos Julgados de Paz
    patterns = {
        "relator": r"Relator:\s*([^\n]+)",
        "data_sentenca": r"Data da Sentença:\s*(\d{2}-\d{2}-\d{4})",
        "julgado_paz_de": r"Julgado de Paz (?:de|do)\s*:\s*([^\n<]+)",  # Capturar após os ":"
    }
    
    # Extrair campos básicos
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key in ["data_sentenca"]:
                try:
                    metadata[key] = datetime.strptime(value, "%d-%m-%Y").strftime("%Y-%m-%d")
                except ValueError:
                    metadata[key] = value  # Manter formato original se não conseguir converter
            else:
                metadata[key] = value

    # Extrair MÚLTIPLOS DESCRITORES
    descritores_match = re.search(
        r"Descritores:\s*([^\n]+(?:\n(?!Data da [Ss]entença|Relator|Decisão|Texto Integral)[^\n:]*)*)", 
        text, 
        re.IGNORECASE | re.MULTILINE
    )
    if descritores_match:
        descritores_text = descritores_match.group(1).strip()
        metadata["descritores"] = clean_descritores(descritores_text)

    # Extrair Decisão Texto Integral
    for element in soup(["script", "style", "nav", "header", "footer", "form"]):
        element.decompose()
    
    full_text = soup.get_text(separator='\n', strip=True)
    
    # Procurar onde começa "Decisão" ou "Texto Integral"
    decisao_match = re.search(
        r'((?:decis[aã]o|texto\s+integral):.*)', 
        full_text, 
        re.IGNORECASE | re.DOTALL
    )
    
    if decisao_match:
        metadata["decisao_texto_integral"] = decisao_match.group(1).strip()
    else:
        # Fallback: usar texto completo
        metadata["decisao_texto_integral"] = full_text

    return metadata

def extract_metadata_from_soup(soup, url, tribunal_nome=None, tipo_direito=None, tipo_caso=None):
    """
    Extrai todos os metadados estruturados do HTML de um acórdão dos Tribunais de Relação.
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

""" TODO: Separar as funções de extração da função de orquestração em docs separados """

# --- Fase 3: Orquestração ---

async def main():
    print("=" * 80)
    print("🏛️  SISTEMA DE SCRAPING DGSI - CASOS LEGAIS".center(80))
    print("=" * 80)
    print("\n📋 Escolha o tipo de pesquisa:")
    print("   1. Tribunais de Relação - Violência Doméstica (TRE, TRL, TRC, TRG, TRP)")
    print("   2. Julgados de Paz - Incumprimento de Contratos")
    
    tipo_escolhido = input("\nEscolha uma opção (1-2): ").strip()
    
    if tipo_escolhido == "1":
        # TRIBUNAIS DE RELAÇÃO - VIOLÊNCIA DOMÉSTICA
        print("\n🏛️ Tribunais de Relação Disponíveis:")
        tribunal_escolhido = input(f"Escolha entre {', '.join(pesquisa_descritores_tr.keys())}: ").strip().upper()
        
        if tribunal_escolhido not in pesquisa_descritores_tr:
            print("❌ Tribunal inválido. A sair.")
            return
        
        url_tribunal = pesquisa_descritores_tr[tribunal_escolhido]
        tribunal_id = f'jtr{tribunal_escolhido[-1].lower()}'  # Ex: "TRL" -> "jtrl"
        tipo_pesquisa = "TR"
        
        print(f"\n🎯 Modo: Tribunal de Relação - {tribunal_escolhido}")
        print(f"📝 Será direcionado para: {url_tribunal}")
        print("💡 Tema: Violência Doméstica")
        
        acordao_links = get_dgsi_acordao_links_multipagina_interactive(tipo_pesquisa, url_tribunal, tribunal_id)
        
        # Configuração para Tribunais de Relação
        tribunal_config = {
            "nome": tribunal_escolhido,
            "tipo_direito": "PENAL",
            "tipo_caso": "VIOLÊNCIA DOMÉSTICA"
        }
        tipo_dados = "TR"
        
    elif tipo_escolhido == "2":
        # JULGADOS DE PAZ - INCUMPRIMENTO DE CONTRATOS
        url_tribunal = pesquisa_julgados_paz
        tribunal_id = "cajp"
        tipo_pesquisa = "JP"
        
        print(f"\n🎯 Modo: Julgados de Paz")
        print(f"📝 Será direcionado para: {url_tribunal}")
        print("💡 Tema: Incumprimento de Contratos")
        
        acordao_links = get_dgsi_acordao_links_multipagina_interactive(tipo_pesquisa, url_tribunal, tribunal_id)
        
        # Configuração para Julgados de Paz
        tribunal_config = None  # Não usa a mesma estrutura
        tipo_dados = "JP"
        
    else:
        print("❌ Opção inválida. A sair.")
        return
    
    if not acordao_links:
        print("❌ Nenhum link foi extraído. A sair.")
        return

    # Fase 2 - Seleção de quantidade e Scraping
    total_links_found = len(acordao_links)
    
    # Perguntar ao utilizador quantos links quer processar
    try:
        response = input(f"\n📊 Foram encontrados {total_links_found} links. Quantos deseja processar? (e.g., 5, 10, ou 'all'): ")
        if response.lower() == 'all':
            num_to_scrape = total_links_found
        else:
            num_to_scrape = int(response)
            if num_to_scrape > total_links_found:
                print(f"⚠️  Aviso: O número pedido ({num_to_scrape}) é maior que o total. A processar todos ({total_links_found}).")
                num_to_scrape = total_links_found
    except (ValueError, TypeError):
        print("⚠️  Entrada inválida. A processar os primeiros 5 links por defeito.")
        num_to_scrape = 5

    links_to_scrape = acordao_links[:num_to_scrape]
    
    print(f"\n🚀 Fase 2: A fazer o scrape de {len(links_to_scrape)} documentos sequencialmente...")
    
    # Execução SEQUENCIAL - processar um URL de cada vez
    results = []
    for i, link in enumerate(links_to_scrape):
        print(f"\n[{i+1}/{len(links_to_scrape)}] 🟢 A processar: {link[:80]}...")
        
        try:
            async with AsyncWebCrawler(headless=True) as crawler:
                result = await crawler.arun(url=link)
            
            if result.status_code == 200:
                soup = BeautifulSoup(result.html, 'html.parser')
                
                # Extrair metadados conforme o tipo
                if tipo_dados == "TR":
                    structured_data = extract_metadata_from_soup(
                        soup, 
                        link, 
                        tribunal_nome=tribunal_config["nome"],
                        tipo_direito=tribunal_config["tipo_direito"],
                        tipo_caso=tribunal_config["tipo_caso"]
                    )
                else:  # JP
                    structured_data = extract_metadata_julgados_paz(soup, link)
                
                print(f"[{i+1}/{len(links_to_scrape)}] ✅ Sucesso!")
                results.append({"status": "success", "data": structured_data})
            else:
                print(f"[{i+1}/{len(links_to_scrape)}] ⚠️  Erro HTTP {result.status_code}")
                results.append({"status": "error", "url": link, "details": f"HTTP {result.status_code}"})
        except Exception as e:
            print(f"[{i+1}/{len(links_to_scrape)}] ❌ Exceção: {repr(e)}")
            results.append({"status": "error", "url": link, "details": repr(e)})
        
        # Pausa entre requests
        await asyncio.sleep(1)
    
    # Fase 3 - Guardar Resultados
    successful_scrapes = [res['data'] for res in results if res and res['status'] == 'success']
    failed_scrapes = [res for res in results if res and res['status'] == 'error']
    
    print("\n" + "=" * 80)
    print(f"✅ SCRAPING COMPLETO".center(80))
    print("=" * 80)
    print(f"✅ Sucessos: {len(successful_scrapes)}")
    print(f"❌ Falhas: {len(failed_scrapes)}")
    
    if successful_scrapes:
        # Nomear arquivo baseado no tipo de dados
        if tipo_dados == "TR":
            output_filename = f"dgsi_tr_violencia_domestica_{tribunal_escolhido.lower()}.json"
        else:  # JP
            output_filename = "dgsi_jp_incumprimento_contratosv2.json"
            
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(successful_scrapes, f, ensure_ascii=False, indent=4)
        print(f"💾 Dados guardados em: {output_filename}")

    if failed_scrapes:
        with open("failed_urls.txt", "w", encoding='utf-8') as f:
            for failure in failed_scrapes:
                f.write(f"{failure['url']} - Motivo: {failure['details']}\n")
        print("⚠️  URLs falhados guardados em: failed_urls.txt")

if __name__ == "__main__":
    asyncio.run(main())
