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

""" ✅ IMPLEMENTADO: Sistema de extração multi-página para pesquisa por descritores.
Agora é possível:
- Extrair links de múltiplas páginas sem fechar o browser
- Navegar interativamente entre páginas de descritores  
- Acumular links com remoção automática de duplicados
- Escolher entre pesquisa livre (modo original) ou por descritores (novo modo)
- Suportar múltiplos formatos de URLs (OpenDocument, IDs hexadecimais, etc.)
- Debug completo para diagnosticar problemas de extração

✅ CORRIGIDO: Problema com formato de links em pesquisa por descritores
- Pesquisa livre: /jtrl.nsf/{ID}?OpenDocument
- Pesquisa descritores: /jtrl.nsf/Por+Ano/?SearchView (página de lista)
- Links de acórdãos: /jtrl.nsf/{ID1}/{ID2}?OpenDocument ou /jtrl.nsf/{ID}
"""

# --- Fase 1: Obtenção de Links (Interativa) ---


pesquisa_tribunais = {
    "TRE": "https://www.dgsi.pt/jtre.nsf/Pesquisa+Livre?OpenForm",  # Tribunal da Relação de Évora
    "TRL": "https://www.dgsi.pt/jtrl.nsf/Pesquisa+Livre?OpenForm",  # Tribunal da Relação de Lisboa
    "TRC": "https://www.dgsi.pt/jtrc.nsf/Pesquisa+Livre?OpenForm",  # Tribunal da Relação de Coimbra
    "TRG": "https://www.dgsi.pt/jtrg.nsf/Pesquisa+Livre?OpenForm",  # Tribunal da Relação de Guimarães
    "TRP": "https://www.dgsi.pt/jtrp.nsf/Pesquisa+Livre?OpenForm",
}  # Tribunal da Relação do Porto

# URLs de pesquisa por descritores (nova funcionalidade)
pesquisa_descritores = {
    "TRE": "https://www.dgsi.pt/jtre.nsf/Pesquisa+Descritor?OpenForm",  # Tribunal da Relação de Évora
    "TRL": "https://www.dgsi.pt/jtrl.nsf/Pesquisa+Descritor?OpenForm",  # Tribunal da Relação de Lisboa
    "TRC": "https://www.dgsi.pt/jtrc.nsf/Pesquisa+Descritor?OpenForm",  # Tribunal da Relação de Coimbra
    "TRG": "https://www.dgsi.pt/jtrg.nsf/Pesquisa+Descritor?OpenForm",  # Tribunal da Relação de Guimarães
    "TRP": "https://www.dgsi.pt/jtrp.nsf/Pesquisa+Descritor?OpenForm",
}  # Tribunal da Relação do Porto


# URL de pesquisa por descritor no julgado de paz


"""Esta função foi substituída pela versão multi-página abaixo. Pode se apagar."""


def get_dgsi_acordao_links_interactive(tribunal, url_tribunal: str):
    """
    Abre um browser para o utilizador fazer a pesquisa manualmente e depois extrai os links da página de resultados.
    VERSÃO ORIGINAL - uma só extração de links.
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
        soup = BeautifulSoup(driver.page_source, "html.parser")

        acordao_links = set()
        for key in pesquisa_tribunais.keys():
            for link in soup.find_all("a", href=True):
                href = link["href"]

                if key == "TRE" and "jtre.nsf" in href:
                    mid_part_url = extract_middle_part_of_url(url_tribunal)
                elif key == "TRL" and "jtrl.nsf" in href:
                    mid_part_url = extract_middle_part_of_url(url_tribunal)
                elif key == "TRC" and "jtrc.nsf" in href:
                    mid_part_url = extract_middle_part_of_url(url_tribunal)
                elif key == "TRG" and "jtrg.nsf" in href:
                    mid_part_url = extract_middle_part_of_url(url_tribunal)
                else:
                    mid_part_url = extract_middle_part_of_url(
                        url_tribunal
                    )  # outputs tribunal do porto

                if "OpenDocument" in href and mid_part_url in href:
                    if not href.startswith("http"):
                        href = "https://www.dgsi.pt" + href
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


def get_dgsi_acordao_links_multipagina_interactive(tribunal, url_tribunal: str):
    """
    NOVA VERSÃO: Mantém o browser aberto para múltiplas extrações de links.
    Ideal para pesquisa por descritores onde o utilizador navega entre várias páginas.
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
        print(
            "   🎯 Este modo permite extrair links de múltiplas páginas sem fechar o browser."
        )
        print(
            "   📝 Ideal para pesquisa por descritores onde navega entre várias páginas de resultados."
        )
        print("\n" + "=" * 80)
        print("⚠️  IMPORTANTE - FLUXO PARA PESQUISA POR DESCRITORES:")
        print("=" * 80)
        print("1. Pesquise por palavras-chave (ex: 'violência doméstica')")
        print("2. Verá uma lista de DESCRITORES possíveis")
        print(
            "3. ⚡ CLIQUE num descritor específico (ex: 'VÍTIMA DE VIOLÊNCIA DOMÉSTICA')"
        )
        print("4. Agora verá a LISTA DE ACÓRDÃOS com esse descritor")
        print(
            "5. NESTA página de lista de acórdãos, volte aqui e pressione '1' para extrair"
        )
        print(
            "6. Se houver múltiplas páginas de resultados, navegue entre elas e repita"
        )
        print("=" * 80 + "\n")

        while True:
            print(f"\n📊 Links acumulados até agora: {len(all_acordao_links)}")
            print("\n🔄 Opções disponíveis:")
            print("   1. Extrair links da página atual")
            print("   2. Terminar e continuar com o scraping")
            print("   3. Sair sem fazer scraping")

            escolha = input("\nEscolha uma opção (1-3): ").strip()

            if escolha == "1":
                print("\n🔍 A extrair links da página atual...")
                print(f"🔍 DEBUG: Tribunal escolhido = {tribunal}")
                print(f"🔍 DEBUG: URL do tribunal = {url_tribunal}")

                soup = BeautifulSoup(driver.page_source, "html.parser")

                # DEBUG: Verificar URL atual no browser
                current_url = driver.current_url
                print(f"🔍 DEBUG: URL atual do browser = {current_url}")

                links_pagina_atual = set()
                all_links_found = []  # Para debug

                # Determinar o identificador do tribunal (ex: "jtrl", "jtrp")
                tribunal_id = f"jtr{tribunal[-1].lower()}"  # Ex: "TRL" -> "jtrl"
                print(f"🔍 DEBUG: Procurando por links com '{tribunal_id}.nsf'")

                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    all_links_found.append(href)  # Guardar todos para debug

                    # Verificar se o link pertence ao tribunal correto
                    if f"{tribunal_id}.nsf" in href.lower():
                        # IMPORTANTE: Na pesquisa por descritores, os links de acórdãos podem ter formatos diferentes:
                        # Formato 1 (comum): /jtrl.nsf/{ID}?OpenDocument
                        # Formato 2 (descritores): /jtrl.nsf/33182fc732316039802568d9003626b5/{ID}?OpenDocument
                        # Formato 3 (vista): /jtrl.nsf/{ID}/hexnumber

                        # Aceitar links que contenham OpenDocument OU que pareçam IDs de documentos
                        is_valid_link = False

                        # Método 1: Links diretos com OpenDocument
                        if "OpenDocument" in href:
                            is_valid_link = True

                        # Método 2: Links com estrutura de ID (hexadecimal de 32 caracteres)
                        # Exemplo: /jtrl.nsf/33182fc732316039802568d9003626b5/abc123...
                        elif re.search(r"/[a-f0-9]{32}/", href, re.IGNORECASE):
                            is_valid_link = True

                        # Método 3: Links simples com ID depois do .nsf/
                        # Exemplo: /jtrl.nsf/abc123def456
                        elif re.match(
                            rf".*{tribunal_id}\.nsf/[a-f0-9]{{32}}", href, re.IGNORECASE
                        ):
                            is_valid_link = True

                        # EXCLUIR: Links de pesquisa/navegação
                        if any(
                            x in href
                            for x in [
                                "SearchView",
                                "CreateDocument",
                                "OpenForm",
                                "Pesquisa",
                                "Por+Ano",
                            ]
                        ):
                            is_valid_link = False

                        if is_valid_link:
                            # Converter para URL absoluto se necessário
                            if not href.startswith("http"):
                                href = "https://www.dgsi.pt" + href
                            links_pagina_atual.add(href)

                # DEBUG: Mostrar estatísticas
                print(f"\n📊 DEBUG: Total de links na página: {len(all_links_found)}")
                print(
                    f"📊 DEBUG: Links do tribunal {tribunal_id}: {sum(1 for h in all_links_found if f'{tribunal_id}.nsf' in h.lower())}"
                )
                print(
                    f"📊 DEBUG: Links com 'OpenDocument': {sum(1 for h in all_links_found if 'OpenDocument' in h)}"
                )
                print(
                    f"📊 DEBUG: Links com IDs hexadecimais: {sum(1 for h in all_links_found if re.search(r'[a-f0-9]{32}', h, re.IGNORECASE))}"
                )

                # Mostrar alguns exemplos de links encontrados do tribunal
                tribunal_links_examples = [
                    h for h in all_links_found if f"{tribunal_id}.nsf" in h.lower()
                ][:10]
                print(
                    f"\n🔍 DEBUG: Primeiros {min(5, len(tribunal_links_examples))} links do {tribunal_id} encontrados:"
                )
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
                    print("\n⚠️  Nenhum link de acórdão encontrado nesta página.")
                    print("💡 POSSÍVEIS CAUSAS:")
                    print(
                        "   1. Está na página de LISTA DE DESCRITORES (precisa clicar num descritor)"
                    )
                    print(
                        "   2. Está na página de pesquisa (precisa fazer a pesquisa primeiro)"
                    )
                    print(
                        "   3. A página ainda está a carregar (aguarde e tente novamente)"
                    )
                    print("   4. Formato de links diferente (veja os exemplos acima)")
                else:
                    print(f"\n✅ Links extraídos com sucesso!")
                    if len(links_novos) > 0:
                        print(f"📝 Exemplos dos novos links encontrados:")
                        for i, link in enumerate(list(links_novos)[:3]):
                            print(f"  {i+1}. {link[:100]}...")

                print(
                    "\n👆 Navegue para a próxima página no browser e volte aqui para extrair mais links."
                )

            elif escolha == "2":
                if len(all_acordao_links) > 0:
                    print(
                        f"\n✅ Finalizando extração com {len(all_acordao_links)} links únicos."
                    )
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
    descritores_raw_split = re.split(r"[;,\n]+", descritores_raw)

    # Lista de padrões a excluir (campos que não são descritores)
    exclude_patterns = [
        r"data\s+do\s+acord[aã]o",
        r"processo:",
        r"relator:",
        r"tribunal:",
        r"secção:",
        r"meio\s+processual:",
        r"decisão:",
        r"votação:",
        r"texto\s+integral",
        r"^[0-9\s\-/]+$",  # apenas números e traços (como datas ou números de processo)
    ]

    valid_descritores = []
    for d in descritores_raw_split:
        d_clean = d.strip()
        if len(d_clean) > 2:  # Mínimo de 3 caracteres
            # Verificar se não corresponde aos padrões de exclusão
            is_excluded = any(
                re.match(pattern, d_clean, re.IGNORECASE)
                for pattern in exclude_patterns
            )
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
    sumario_tags = soup.find_all(
        ["b", "strong"], string=re.compile(r"sum[aá]rio", re.IGNORECASE)
    )

    for tag in sumario_tags:
        content_parts = []

        # Buscar conteúdo nos siblings seguintes
        for sibling in tag.find_next_siblings():
            if not sibling:
                continue

            # Parar se encontrar outros títulos principais
            if sibling.name in ["b", "strong"] and re.search(
                r"(decis[aã]o|acordam|relatório|voto|fundamentos)",
                sibling.get_text(),
                re.IGNORECASE,
            ):
                break

            # Extrair texto do sibling
            text = (
                sibling.get_text(strip=True)
                if hasattr(sibling, "get_text")
                else str(sibling).strip()
            )
            if text and len(text) > 10:  # Ignorar textos muito curtos
                content_parts.append(text)

        if content_parts:
            sumario_text = "\n".join(content_parts).strip()
            break

    # Estratégia 2: Buscar por regex no texto completo se Estratégia 1 falhou
    if not sumario_text:
        full_text = soup.get_text(separator="\n", strip=True)

        # Padrão específico: capturar tudo entre "Sumário:" e "Decisão Texto Integral/Parcial:"
        sumario_match = re.search(
            r"sum[aá]rio:?\s*\n?(.*?)(?=\n\s*(?:decis[aã]o\s+texto\s+(?:integral|parcial)|decis[aã]o|relatório|acordam|fundamentos|voto):|$)",
            full_text,
            re.IGNORECASE | re.DOTALL,
        )

        if sumario_match:
            sumario_text = sumario_match.group(1).strip()

    # Estratégia 3: Buscar em estrutura de tabela (comum no DGSI)
    if not sumario_text:
        table_cells = soup.find_all(["td", "th"])
        for cell in table_cells:
            cell_text = cell.get_text(strip=True)
            if re.match(r"sum[aá]rio:?", cell_text, re.IGNORECASE):
                # Buscar célula seguinte ou próximo sibling
                next_cell = cell.find_next_sibling(["td", "th"])
                if next_cell:
                    sumario_text = next_cell.get_text(strip=True)
                    break

    # Limpeza final e validação
    if sumario_text:
        # Remover linhas muito curtas e limpeza geral
        lines = [line.strip() for line in sumario_text.split("\n") if line.strip()]
        clean_lines = [
            line for line in lines if len(line) > 5
        ]  # Remover linhas muito curtas

        if clean_lines:
            sumario_text = "\n".join(clean_lines)

            # Validação: sumário deve ter pelo menos 50 caracteres para ser válido
            if len(sumario_text) >= 50:
                return sumario_text

    return None


# --- Fase 2: Scraping e Extração Estruturada ---


def extract_metadata_from_soup(
    soup, url, tribunal_nome=None, tipo_direito=None, tipo_caso=None
):
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
        "campos_especificos": {},
    }

    # Extração baseada em texto e padrões (regex)
    text = soup.get_text(separator="\n", strip=True)

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
                    metadata[key] = datetime.strptime(value, "%d-%m-%Y").strftime(
                        "%Y-%m-%d"
                    )
                except ValueError:
                    pass  # Deixa como None se o formato for inesperado
            elif key == "texto_integral_disponivel":
                metadata[key] = (
                    value.upper() == "S"
                )  # Boolean: True se "S", False se "N"
            else:
                metadata[key] = value

    # Extrair sumário usando função avançada dedicada
    metadata["sumario"] = extract_sumario_advanced(soup)

    # Extrair MÚLTIPLOS DESCRITORES (melhor parsing)
    descritores_match = re.search(
        r"Descritores:\s*([^\n]+(?:\n[^\n:]*)*)", text, re.IGNORECASE | re.MULTILINE
    )
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

    full_text = soup.get_text(separator="\n", strip=True)

    # Procurar onde começa "Decisão Texto Integral:"
    decisao_match = re.search(
        r"(decis[aã]o\s+texto\s+integral:.*)", full_text, re.IGNORECASE | re.DOTALL
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
            "tipo_caso": "VIOLÊNCIA DOMÉSTICA",
        }

    try:
        async with AsyncWebCrawler(headless=True) as crawler:
            result = await crawler.arun(url=url)

        if result.status_code == 200:
            soup = BeautifulSoup(result.html, "html.parser")
            structured_data = extract_metadata_from_soup(
                soup,
                url,
                tribunal_nome=tribunal_config["nome"],
                tipo_direito=tribunal_config["tipo_direito"],
                tipo_caso=tribunal_config["tipo_caso"],
            )
            print(f"[{session_id}/{total}] ✅ Sucesso: {url[:80]}")
            return {"status": "success", "data": structured_data}
        else:
            print(
                f"[{session_id}/{total}] ⚠️ Erro HTTP {result.status_code}: {url[:80]}"
            )
            return {
                "status": "error",
                "url": url,
                "details": f"HTTP status {result.status_code}",
            }
    except Exception as e:
        error_details = repr(e)  # Usar repr() para obter mais detalhes do erro
        print(f"[{session_id}/{total}] ❌ Exceção: {url[:80]} - {error_details}")
        return {"status": "error", "url": url, "details": error_details}


""" TODO: Separar as funções de extração da função de orquestração em docs separados """

# --- Fase 3: Orquestração ---


async def main():
    # Perguntar ao utilizador qual o tribunal a pesquisar
    print("🏛️ Escolha o tribunal para a pesquisa:")
    tribunal_escolhido = (
        input(f"Escolha entre {','.join(pesquisa_tribunais.keys())}:").strip().upper()
    )

    if tribunal_escolhido not in pesquisa_tribunais:
        print("❌ Tribunal inválido. A sair.")
        return

    # Escolher tipo de pesquisa
    print(f"\n📋 Escolha o tipo de pesquisa para {tribunal_escolhido}:")
    print("   1. Pesquisa Livre (modo original - uma só página)")
    print("   2. Pesquisa por Descritores (modo multi-página)")

    tipo_pesquisa = input("\nEscolha o tipo de pesquisa (1-2): ").strip()

    if tipo_pesquisa == "1":
        # Modo original - pesquisa livre
        url_tribunal = pesquisa_tribunais[tribunal_escolhido]
        acordao_links = get_dgsi_acordao_links_interactive(
            tribunal_escolhido, url_tribunal=url_tribunal
        )
    elif tipo_pesquisa == "2":
        # Novo modo - pesquisa por descritores (multi-página)
        url_tribunal = pesquisa_descritores[tribunal_escolhido]
        print(f"\n🎯 Modo Pesquisa por Descritores ativado!")
        print(f"📝 Será direcionado para: {url_tribunal}")
        print(
            "💡 Dica: Pesquise por palavras-chave como 'violência doméstica', 'divórcio', etc."
        )
        print(
            "🔄 Poderá navegar entre múltiplas páginas de descritores e extrair links de todas."
        )

        acordao_links = get_dgsi_acordao_links_multipagina_interactive(
            tribunal_escolhido, url_tribunal=url_tribunal
        )
    else:
        print("❌ Opção inválida. A sair.")
        return

    if not acordao_links:
        return

    # Fase 2 - Seleção de quantidade e Scraping
    total_links_found = len(acordao_links)

    # Perguntar ao utilizador quantos links quer processar
    try:
        response = input(
            f"\nForam encontrados {total_links_found} links. Quantos deseja processar? (e.g., 5, 10, ou 'all'): "
        )
        if response.lower() == "all":
            num_to_scrape = total_links_found
        else:
            num_to_scrape = int(response)
            if num_to_scrape > total_links_found:
                print(
                    f"Aviso: O número pedido ({num_to_scrape}) é maior que o total de links encontrados ({total_links_found}). A processar todos."
                )
                num_to_scrape = total_links_found
    except (ValueError, TypeError):
        print("Entrada inválida. A processar os primeiros 5 links por defeito.")
        num_to_scrape = 5

    links_to_scrape = acordao_links[:num_to_scrape]

    # Configuração manual do tribunal (flexível)
    """ TODO: Melhorar esta parte para evitar input manual repetido """

    print(f"\n🏛️ Configuração do Tribunal :")
    tribunal_nome = (
        input(
            "Nome do tribunal (choose between: TRP, TRL, TRC, TRG, TRE) [padrão: TRP]: "
        ).strip()
        or "TRP"
    )
    tipo_direito = (
        input(
            "Tipo de direito (choose between: PENAL, CÍVEL, TRABALHO ) [padrão: PENAL]: "
        ).strip()
        or "PENAL"
    )
    tipo_caso = (
        input(
            "Tipo de caso (ex: VIOLÊNCIA DOMÉSTICA) [padrão: VIOLÊNCIA DOMÉSTICA]: "
        ).strip()
        or "VIOLÊNCIA DOMÉSTICA"
    )

    tribunal_config = {
        "nome": tribunal_nome,
        "tipo_direito": tipo_direito,
        "tipo_caso": tipo_caso,
    }

    print(
        f"\nFase 2: A fazer o scrape de {len(links_to_scrape)} acórdãos sequencialmente (um de cada vez)..."
    )

    # Execução SEQUENCIAL - processar um URL de cada vez
    results = []
    for i, link in enumerate(links_to_scrape):
        result = await scrape_and_structure_acordao(
            link, i + 1, len(links_to_scrape), tribunal_config
        )
        results.append(result)

        # Pequena pausa entre requests para ser gentil com o servidor
        await asyncio.sleep(1)

    # Fase 3
    successful_scrapes = [
        res["data"] for res in results if res and res["status"] == "success"
    ]
    failed_scrapes = [res for res in results if res and res["status"] == "error"]

    print(
        f"\n✅ Scraping completo. Sucessos: {len(successful_scrapes)}, Falhas: {len(failed_scrapes)}"
    )

    if successful_scrapes:
        # Nomear arquivo baseado no tipo de pesquisa
        if tipo_pesquisa == "2":
            output_filename = "dgsi_acordaos_descritores_v1.json"
        else:
            output_filename = "dgsi_acordaos_structured_v6.json"

        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(successful_scrapes, f, ensure_ascii=False, indent=4)
        print(f"💾 Todos os dados estruturados foram guardados em {output_filename}")

    if failed_scrapes:
        with open("failed_urls.txt", "w", encoding="utf-8") as f:
            for failure in failed_scrapes:
                f.write(f"{failure['url']} - Motivo: {failure['details']}\n")
        print("⚠️ Alguns URLs falharam. Verifique failed_urls.txt para mais detalhes.")


if __name__ == "__main__":
    asyncio.run(main())
