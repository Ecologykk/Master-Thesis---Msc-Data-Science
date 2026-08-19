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


def extract_decisao_from_text(texto_integral_completo):
    """
    Extrai a decisão final do texto integral focando APENAS NO FIM do documento.
    
    ESTRATÉGIA v3 (FOCADA NO FIM):
    - A decisão SEMPRE está no final do documento (últimos 30-40%)
    - Aplicamos TODOS os padrões APENAS nessa região final
    - Ignoramos completamente os primeiros 60-70% (Relatório, Fundamentação)
    
    PRIORIDADES (aplicadas aos últimos 30-40%):
    1. DISPOSITIVO com seção numerada (IV-VI)
    2. DECISÃO com seção numerada (III-VI)
    3. "Pelo exposto" + verbos decisórios próximos
    4. "Acordam" (última tentativa, com flag)
    5. Edge case → verificação manual
    
    Args:
        texto_integral_completo: Texto completo do acórdão
        
    Returns:
        dict com: decisao_extraida, texto_sem_decisao, metadata_decisao
    """
    
    if not texto_integral_completo or len(texto_integral_completo) < 100:
        return {
            "decisao_extraida_do_texto_integral": None,
            "texto_integral_sem_decisao": texto_integral_completo if texto_integral_completo else "",
            "metadata_decisao": {
                "metodo_extracao": "none",
                "confianca": "baixa",
                "palavra_chave_encontrada": None,
                "requer_verificacao_manual": True,
                "motivo_verificacao": "Texto integral muito curto ou vazio"
            }
        }
    
    # ========================================================================
    # PASSO 1: THRESHOLD ADAPTATIVO - buscar nos últimos 30%, 40% ou 50%
    # ========================================================================
    # Para documentos longos com referências no final, precisamos de threshold adaptativo
    # Tentamos múltiplos thresholds: 70% (buscar nos últimos 30%), depois 60%, depois 50%
    
    def try_extraction_with_threshold(threshold_pct):
        """
        Tenta extrair decisão começando num threshold específico.
        Retorna (sucesso, resultado) onde sucesso indica se encontrou algo válido.
        """
        texto_length = len(texto_integral_completo)
        inicio_busca = int(texto_length * threshold_pct)
        texto_final = texto_integral_completo[inicio_busca:]
        
        # Retornar o texto final e posição para processamento
        return inicio_busca, texto_final
    
    # Começamos com 70% (últimos 30%) - padrão
    thresholds_to_try = [0.70, 0.60, 0.50]  # 70%, 60%, 50%
    texto_final = None
    inicio_busca = None
    
    # Por agora, usar apenas o primeiro threshold (70%)
    # Se não encontrar nada válido nas camadas, o sistema tentará os outros
    texto_length = len(texto_integral_completo)
    inicio_busca = int(texto_length * 0.70)
    texto_final = texto_integral_completo[inicio_busca:]
    
    # Padrões regex ROBUSTOS
    # Lista de palavras-chave judiciais expandida (decisão final do tribunal)
    PALAVRAS_DECISAO = r'(?:DISPOSITIVO|Dispositivo|DECISÃO|Decisão|CONCLUSÃO|Conclusão|DELIBERAÇÃO|Deliberação|SENTENÇA|Sentença|VEREDICTO|Veredicto)'
    
    # DISPOSITIVO/DECISÃO com seção numerada ROMANA (padrão preferencial)
    # Nota: [\s.\-–—:]* aceita espaços, ponto, hífen, en-dash (–), em-dash (—), dois-pontos
    # Expandido para III-X (antes só tinha III-VI) para capturar "VII- Decisão", etc.
    DISPOSITIVO_PATTERN = rf'(?:^|\n)\s*((?:III|IV|V|VI|VII|VIII|IX|X)\s*[\s.\-–—:]*\s*{PALAVRAS_DECISAO})[:.\s]*'
    DECISAO_PATTERN = rf'(?:^|\n)\s*((?:III|IV|V|VI|VII|VIII|IX|X)\s*[\s.\-–—:]*\s*{PALAVRAS_DECISAO})[:.\s]*'
    
    # DISPOSITIVO/DECISÃO com seção ALFABÉTICA (A-F) - expandido para incluir A e B
    # Exemplos: "A. DECISÃO", "C - Dispositivo", "D. Decisão", "E – Conclusão"
    DISPOSITIVO_ALFABETICO_PATTERN = rf'(?:^|\n)\s*((?:[A-F])\s*[\s.\-–—:]*\s*{PALAVRAS_DECISAO})[:.\s]*'
    DECISAO_ALFABETICO_PATTERN = rf'(?:^|\n)\s*((?:[A-F])\s*[\s.\-–—:]*\s*{PALAVRAS_DECISAO})[:.\s]*'
    
    # DISPOSITIVO/DECISÃO SEM numeração (CAPS LOCK) - para juízes que não numeram
    DISPOSITIVO_CAPS_PATTERN = r'(?:^|\n)\s*(DISPOSITIVO|DECISÃO|CONCLUSÃO|DELIBERAÇÃO)[:.\s]*'
    DECISAO_CAPS_PATTERN = r'(?:^|\n)\s*(DECISÃO|DISPOSITIVO|CONCLUSÃO|DELIBERAÇÃO|FUNDAMENTAÇÃO\s+DA\s+DECISÃO)[:.\s]*'
    
    # NUMERAÇÃO DECIMAL COM NOME (ex: "3. DECISÃO", "3.Decisão") - prioridade sobre isolada
    NUMERACAO_DECIMAL_COM_NOME_PATTERN = rf'(?:^|\n)\s*([3-9]|[1-9][0-9])\s*[\s.\-–—:]*\s*{PALAVRAS_DECISAO}'
    
    # NUMERAÇÃO ALFABÉTICA COM NOME (ex: "A. DECISÃO", "C. DECISÃO", "D - Dispositivo") - expandido A-F
    NUMERACAO_ALFABETICA_COM_NOME_PATTERN = rf'(?:^|\n)\s*([A-F])\s*[\s.\-–—:]*\s*{PALAVRAS_DECISAO}'
    
    # NUMERAÇÃO ISOLADA (apenas números/letras sem nomes de secções)
    # Romana isolada: III-X (expandido de III-VI)
    NUMERACAO_ROMANA_ISOLADA_PATTERN = r'(?:^|\n)\s*((?:III|IV|V|VI|VII|VIII|IX|X))[\s.\-–—:]*(?=\n|$)'
    # Alfabética isolada: A-F (expandido de C-F)
    NUMERACAO_ALFABETICA_ISOLADA_PATTERN = r'(?:^|\n)\s*([A-F])[\s.\-–—:]*(?=\n|$)'
    # Decimal: 3., 4., 5., etc. seguido de espaço/quebra (início de secção)
    NUMERACAO_DECIMAL_PATTERN = r'(?:^|\n)\s*([3-9]|[1-9][0-9])\s*[\s.\-–—:]*\s*(?:\n|[A-ZÀ-Ú])'
    
    # Expressões de conclusão (expandidas para capturar mais variações)
    # Adicionado: "Termos em que acordam", "Nos termos expostos"
    PELO_EXPOSTO_PATTERN = r'(Pelo|Face ao|Nestes|Nos)\s+(?:exposto|termos|que precede|quanto|termos\s+expostos|termos\s+em\s+que\s+(?:se\s+)?(?:decide|acordam))s?[,:]'
    
    # Acordam (expandido para capturar "acorda-se" e mais contextos)
    # Adicionado: "acorda-se", contextos como "em negar", "em conceder"
    ACORDAM_PATTERN = r'(Acordam|Acordaram|acorda-se)\s+(?:os\s+[Jj]u[ií]zes|em\s+conferência|no\s+Tribunal|em\s+(?:negar|conceder|julgar|manter|revogar))?'
    
    # Verbos decisórios expandidos
    VERBOS_DECISORIOS = r'\b(julga-se|condena-se|absolve-se|confirma-se|anula-se|revoga-se|indefere-se|mantém-se|decide-se|determina-se|nega-se|negar|negado\s+provimento|nega-se\s+provimento|dá-se|concede-se|procede|improcede|procedente|improcedente|manter|confirmar|revogar)\b'
    
    resultado = {
        "decisao_extraida_do_texto_integral": None,
        "texto_integral_sem_decisao": texto_integral_completo,  # Default: texto completo
        "metadata_decisao": {
            "metodo_extracao": None,
            "confianca": None,
            "palavra_chave_encontrada": None,
            "requer_verificacao_manual": False,
            "motivo_verificacao": None
        }
    }
    
    # ========================================================================
    # CAMADA 1A: DISPOSITIVO com numeração ROMANA (prioridade máxima)
    # ========================================================================
    dispositivo_matches = list(re.finditer(DISPOSITIVO_PATTERN, texto_final, re.IGNORECASE | re.MULTILINE))
    
    if dispositivo_matches:
        first_dispositivo = dispositivo_matches[0]
        start_pos_absolute = inicio_busca + first_dispositivo.start()
        decisao_text = texto_integral_completo[start_pos_absolute:].strip()
        has_verbos = re.search(VERBOS_DECISORIOS, decisao_text[:1000], re.IGNORECASE)
        
        if has_verbos and 50 < len(decisao_text) < 50000:
            resultado["decisao_extraida_do_texto_integral"] = decisao_text
            resultado["texto_integral_sem_decisao"] = texto_integral_completo[:start_pos_absolute].strip()
            resultado["metadata_decisao"] = {
                "metodo_extracao": "dispositivo_numerado_romano",
                "confianca": "alta",
                "palavra_chave_encontrada": first_dispositivo.group(1).strip(),
                "requer_verificacao_manual": False,
                "motivo_verificacao": None
            }
            return resultado
    
    # ========================================================================
    # CAMADA 1A-BIS: DISPOSITIVO com numeração ALFABÉTICA (C, D, E, F) - NOVO
    # ========================================================================
    dispositivo_alfa_matches = list(re.finditer(DISPOSITIVO_ALFABETICO_PATTERN, texto_final, re.IGNORECASE | re.MULTILINE))
    
    if dispositivo_alfa_matches:
        first_dispositivo_alfa = dispositivo_alfa_matches[0]
        start_pos_absolute = inicio_busca + first_dispositivo_alfa.start()
        decisao_text = texto_integral_completo[start_pos_absolute:].strip()
        has_verbos = re.search(VERBOS_DECISORIOS, decisao_text[:1000], re.IGNORECASE)
        
        if has_verbos and 50 < len(decisao_text) < 50000:
            resultado["decisao_extraida_do_texto_integral"] = decisao_text
            resultado["texto_integral_sem_decisao"] = texto_integral_completo[:start_pos_absolute].strip()
            resultado["metadata_decisao"] = {
                "metodo_extracao": "dispositivo_alfabetico",
                "confianca": "alta",
                "palavra_chave_encontrada": first_dispositivo_alfa.group(1).strip(),
                "requer_verificacao_manual": False,
                "motivo_verificacao": None
            }
            return resultado
    
    # ========================================================================
    # CAMADA 1B: DISPOSITIVO SEM numeração (CAPS LOCK) - juízes que não numeram
    # ========================================================================
    dispositivo_caps_matches = list(re.finditer(DISPOSITIVO_CAPS_PATTERN, texto_final, re.MULTILINE))
    
    if dispositivo_caps_matches:
        first_dispositivo_caps = dispositivo_caps_matches[0]
        start_pos_absolute = inicio_busca + first_dispositivo_caps.start()
        decisao_text = texto_integral_completo[start_pos_absolute:].strip()
        has_verbos = re.search(VERBOS_DECISORIOS, decisao_text[:1000], re.IGNORECASE)
        
        if has_verbos and 50 < len(decisao_text) < 50000:
            resultado["decisao_extraida_do_texto_integral"] = decisao_text
            resultado["texto_integral_sem_decisao"] = texto_integral_completo[:start_pos_absolute].strip()
            resultado["metadata_decisao"] = {
                "metodo_extracao": "dispositivo_caps",
                "confianca": "alta",
                "palavra_chave_encontrada": first_dispositivo_caps.group(1).strip(),
                "requer_verificacao_manual": False,
                "motivo_verificacao": None
            }
            return resultado
    
    # ========================================================================
    # CAMADA 2A: DECISÃO com numeração ROMANA
    # ========================================================================
    decisao_matches = list(re.finditer(DECISAO_PATTERN, texto_final, re.IGNORECASE | re.MULTILINE))
    
    if decisao_matches:
        first_decisao = decisao_matches[0]
        start_pos_absolute = inicio_busca + first_decisao.start()
        decisao_text = texto_integral_completo[start_pos_absolute:].strip()
        
        # Verificar se há DISPOSITIVO depois (qualquer tipo)
        dispositivo_after = re.search(DISPOSITIVO_PATTERN, decisao_text, re.IGNORECASE | re.MULTILINE)
        if not dispositivo_after:
            dispositivo_after = re.search(DISPOSITIVO_ALFABETICO_PATTERN, decisao_text, re.IGNORECASE | re.MULTILINE)
        if not dispositivo_after:
            dispositivo_after = re.search(DISPOSITIVO_CAPS_PATTERN, decisao_text, re.MULTILINE)
        
        if dispositivo_after:
            dispositivo_start_absolute = start_pos_absolute + dispositivo_after.start()
            decisao_text = texto_integral_completo[dispositivo_start_absolute:].strip()
            texto_sem_decisao = texto_integral_completo[:dispositivo_start_absolute].strip()
            metodo = "dispositivo_romano_ou_caps"
            confianca = "alta"
            palavra_chave = dispositivo_after.group(1).strip()
            requer_verificacao = False
        else:
            has_verbos = re.search(VERBOS_DECISORIOS, decisao_text[:1000], re.IGNORECASE)
            texto_sem_decisao = texto_integral_completo[:start_pos_absolute].strip()
            metodo = "decisao_numerada_romana"
            confianca = "media" if has_verbos else "baixa"
            palavra_chave = first_decisao.group(1).strip()
            requer_verificacao = (confianca == "baixa")
        
        if 50 < len(decisao_text) < 50000:
            resultado["decisao_extraida_do_texto_integral"] = decisao_text
            resultado["texto_integral_sem_decisao"] = texto_sem_decisao
            resultado["metadata_decisao"] = {
                "metodo_extracao": metodo,
                "confianca": confianca,
                "palavra_chave_encontrada": palavra_chave,
                "requer_verificacao_manual": requer_verificacao,
                "motivo_verificacao": "Decisão sem verbos decisórios evidentes" if requer_verificacao else None
            }
            return resultado
    
    # ========================================================================
    # CAMADA 2A-BIS: DECISÃO com numeração ALFABÉTICA (C, D, E, F) - NOVO
    # ========================================================================
    decisao_alfa_matches = list(re.finditer(DECISAO_ALFABETICO_PATTERN, texto_final, re.IGNORECASE | re.MULTILINE))
    
    if decisao_alfa_matches:
        first_decisao_alfa = decisao_alfa_matches[0]
        start_pos_absolute = inicio_busca + first_decisao_alfa.start()
        decisao_text = texto_integral_completo[start_pos_absolute:].strip()
        
        # Verificar se há DISPOSITIVO depois (qualquer tipo)
        dispositivo_after = re.search(DISPOSITIVO_ALFABETICO_PATTERN, decisao_text, re.IGNORECASE | re.MULTILINE)
        if not dispositivo_after:
            dispositivo_after = re.search(DISPOSITIVO_PATTERN, decisao_text, re.IGNORECASE | re.MULTILINE)
        if not dispositivo_after:
            dispositivo_after = re.search(DISPOSITIVO_CAPS_PATTERN, decisao_text, re.MULTILINE)
        
        if dispositivo_after:
            dispositivo_start_absolute = start_pos_absolute + dispositivo_after.start()
            decisao_text = texto_integral_completo[dispositivo_start_absolute:].strip()
            texto_sem_decisao = texto_integral_completo[:dispositivo_start_absolute].strip()
            metodo = "dispositivo_alfabetico_ou_outro"
            confianca = "alta"
            palavra_chave = dispositivo_after.group(1).strip()
            requer_verificacao = False
        else:
            has_verbos = re.search(VERBOS_DECISORIOS, decisao_text[:1000], re.IGNORECASE)
            texto_sem_decisao = texto_integral_completo[:start_pos_absolute].strip()
            metodo = "decisao_numerada_alfabetica"
            confianca = "media" if has_verbos else "baixa"
            palavra_chave = first_decisao_alfa.group(1).strip()
            requer_verificacao = (confianca == "baixa")
        
        if 50 < len(decisao_text) < 50000:
            resultado["decisao_extraida_do_texto_integral"] = decisao_text
            resultado["texto_integral_sem_decisao"] = texto_sem_decisao
            resultado["metadata_decisao"] = {
                "metodo_extracao": metodo,
                "confianca": confianca,
                "palavra_chave_encontrada": palavra_chave,
                "requer_verificacao_manual": requer_verificacao,
                "motivo_verificacao": "Decisão sem verbos decisórios evidentes" if requer_verificacao else None
            }
            return resultado
    
    # ========================================================================
    # CAMADA 2B: DECISÃO SEM numeração (CAPS LOCK)
    # ========================================================================
    decisao_caps_matches = list(re.finditer(DECISAO_CAPS_PATTERN, texto_final, re.MULTILINE))
    
    if decisao_caps_matches:
        first_decisao_caps = decisao_caps_matches[0]
        start_pos_absolute = inicio_busca + first_decisao_caps.start()
        decisao_text = texto_integral_completo[start_pos_absolute:].strip()
        
        # Verificar se há DISPOSITIVO depois
        dispositivo_after = re.search(DISPOSITIVO_CAPS_PATTERN, decisao_text, re.MULTILINE)
        
        if dispositivo_after and dispositivo_after.group(1) != first_decisao_caps.group(1):
            dispositivo_start_absolute = start_pos_absolute + dispositivo_after.start()
            decisao_text = texto_integral_completo[dispositivo_start_absolute:].strip()
            texto_sem_decisao = texto_integral_completo[:dispositivo_start_absolute].strip()
            metodo = "dispositivo_caps"
            confianca = "alta"
            palavra_chave = dispositivo_after.group(1).strip()
            requer_verificacao = False
        else:
            has_verbos = re.search(VERBOS_DECISORIOS, decisao_text[:1000], re.IGNORECASE)
            texto_sem_decisao = texto_integral_completo[:start_pos_absolute].strip()
            metodo = "decisao_caps"
            confianca = "media" if has_verbos else "baixa"
            palavra_chave = first_decisao_caps.group(1).strip()
            requer_verificacao = (confianca == "baixa")
        
        if 50 < len(decisao_text) < 50000:
            resultado["decisao_extraida_do_texto_integral"] = decisao_text
            resultado["texto_integral_sem_decisao"] = texto_sem_decisao
            resultado["metadata_decisao"] = {
                "metodo_extracao": metodo,
                "confianca": confianca,
                "palavra_chave_encontrada": palavra_chave,
                "requer_verificacao_manual": requer_verificacao,
                "motivo_verificacao": "Decisão sem verbos decisórios evidentes" if requer_verificacao else None
            }
            return resultado
    
    # ========================================================================
    # CAMADA 2.5: NUMERAÇÃO DECIMAL COM NOME - última ocorrência (3. DECISÃO, 4.Decisão, etc.)
    # ========================================================================
    # Estratégia: capturar "3. DECISÃO" ou "3.Decisão" explicitamente
    # Prioridade ANTES da numeração isolada para evitar falsos positivos
    numeracao_decimal_nome_matches = list(re.finditer(NUMERACAO_DECIMAL_COM_NOME_PATTERN, texto_final, re.IGNORECASE | re.MULTILINE))
    
    if numeracao_decimal_nome_matches:
        # Usar a ÚLTIMA ocorrência (mais provável ser a decisão)
        last_match = numeracao_decimal_nome_matches[-1]
        start_pos_absolute = inicio_busca + last_match.start()
        decisao_text = texto_integral_completo[start_pos_absolute:].strip()
        
        # Validação: verificar se tem conteúdo decisório
        has_verbos = re.search(VERBOS_DECISORIOS, decisao_text[:1000], re.IGNORECASE)
        has_pelo_exposto = re.search(PELO_EXPOSTO_PATTERN, decisao_text[:500], re.IGNORECASE)
        
        if (has_verbos or has_pelo_exposto) and 50 < len(decisao_text) < 50000:
            resultado["decisao_extraida_do_texto_integral"] = decisao_text
            resultado["texto_integral_sem_decisao"] = texto_integral_completo[:start_pos_absolute].strip()
            resultado["metadata_decisao"] = {
                "metodo_extracao": "numeracao_decimal_com_nome",
                "confianca": "alta",
                "palavra_chave_encontrada": f"{last_match.group(1)}.{last_match.group(2)}",
                "requer_verificacao_manual": False,
                "motivo_verificacao": None
            }
            return resultado
    
    # ========================================================================
    # CAMADA 2.5-BIS: NUMERAÇÃO ALFABÉTICA COM NOME - última ocorrência (C. DECISÃO, D - Dispositivo, etc.) - NOVO
    # ========================================================================
    # Estratégia: capturar "C. DECISÃO", "D - Dispositivo", "E – Conclusão" explicitamente
    numeracao_alfa_nome_matches = list(re.finditer(NUMERACAO_ALFABETICA_COM_NOME_PATTERN, texto_final, re.IGNORECASE | re.MULTILINE))
    
    if numeracao_alfa_nome_matches:
        # Usar a ÚLTIMA ocorrência (mais provável ser a decisão)
        last_match = numeracao_alfa_nome_matches[-1]
        start_pos_absolute = inicio_busca + last_match.start()
        decisao_text = texto_integral_completo[start_pos_absolute:].strip()
        
        # Validação: verificar se tem conteúdo decisório
        has_verbos = re.search(VERBOS_DECISORIOS, decisao_text[:1000], re.IGNORECASE)
        has_pelo_exposto = re.search(PELO_EXPOSTO_PATTERN, decisao_text[:500], re.IGNORECASE)
        
        if (has_verbos or has_pelo_exposto) and 50 < len(decisao_text) < 50000:
            resultado["decisao_extraida_do_texto_integral"] = decisao_text
            resultado["texto_integral_sem_decisao"] = texto_integral_completo[:start_pos_absolute].strip()
            resultado["metadata_decisao"] = {
                "metodo_extracao": "numeracao_alfabetica_com_nome",
                "confianca": "alta",
                "palavra_chave_encontrada": last_match.group(0).strip(),
                "requer_verificacao_manual": False,
                "motivo_verificacao": None
            }
            return resultado
    
    # ========================================================================
    # CAMADA 3: "Pelo exposto" / "Termos em que" + verbos decisórios
    # NOTA: Movido para ANTES da numeração isolada para evitar falsos positivos
    # em listas de factos numeradas (ex: "82 - O Arguido agiu...")
    # ========================================================================
    pelo_exposto_matches = list(re.finditer(PELO_EXPOSTO_PATTERN, texto_final, re.IGNORECASE))
    
    if pelo_exposto_matches:
        # Usar a PRIMEIRA ocorrência encontrada no final
        first_match = pelo_exposto_matches[0]
        start_pos_absolute = inicio_busca + first_match.start()
        decisao_text = texto_integral_completo[start_pos_absolute:].strip()
        
        # VALIDAÇÃO: verbos decisórios devem estar nos próximos 300 chars
        snippet = decisao_text[:300]
        has_verbos = re.search(VERBOS_DECISORIOS, snippet, re.IGNORECASE)
        
        if has_verbos and 50 < len(decisao_text) < 50000:
            resultado["decisao_extraida_do_texto_integral"] = decisao_text
            resultado["texto_integral_sem_decisao"] = texto_integral_completo[:start_pos_absolute].strip()
            resultado["metadata_decisao"] = {
                "metodo_extracao": "pelo_exposto",
                "confianca": "media-alta",
                "palavra_chave_encontrada": first_match.group(0).strip(),
                "requer_verificacao_manual": False,
                "motivo_verificacao": None
            }
            return resultado
    
    # ========================================================================
    # CAMADA 4: "Acordam" - expressões de deliberação colegiada
    # NOTA: Movido para ANTES da numeração isolada
    # ========================================================================
    acordam_matches = list(re.finditer(ACORDAM_PATTERN, texto_final, re.IGNORECASE))
    
    if acordam_matches:
        # Usar a PRIMEIRA ocorrência encontrada no final
        first_match = acordam_matches[0]
        start_pos_absolute = inicio_busca + first_match.start()
        decisao_text = texto_integral_completo[start_pos_absolute:].strip()
        
        # Validar verbos decisórios próximos
        has_verbos = re.search(VERBOS_DECISORIOS, decisao_text[:400], re.IGNORECASE)
        
        if 50 < len(decisao_text) < 50000:
            resultado["decisao_extraida_do_texto_integral"] = decisao_text
            resultado["texto_integral_sem_decisao"] = texto_integral_completo[:start_pos_absolute].strip()
            resultado["metadata_decisao"] = {
                "metodo_extracao": "acordam_final",
                "confianca": "media" if has_verbos else "media-baixa",
                "palavra_chave_encontrada": first_match.group(1).strip(),
                "requer_verificacao_manual": not has_verbos,
                "motivo_verificacao": "Extração baseada em 'Acordam' - validar manualmente" if not has_verbos else None
            }
            return resultado
    
    # ========================================================================
    # CAMADA 5A: NUMERAÇÃO ROMANA ISOLADA - última ocorrência (III-X)
    # NOTA: Movido para DEPOIS de "Pelo exposto" e "Acordam" para evitar
    # capturar numerações de listas de factos probatórios
    # ========================================================================
    # Estratégia: alguns juízes usam APENAS numeração romana sem nomes de secções
    # Exemplo: "III" isolado, seguido de texto da decisão
    # Procuramos a ÚLTIMA ocorrência de numeração romana alta (III-X) no final
    numeracao_romana_matches = list(re.finditer(NUMERACAO_ROMANA_ISOLADA_PATTERN, texto_final, re.MULTILINE))
    
    if numeracao_romana_matches:
        # Usar a ÚLTIMA ocorrência (mais provável ser a decisão)
        last_match = numeracao_romana_matches[-1]
        start_pos_absolute = inicio_busca + last_match.start()
        decisao_text = texto_integral_completo[start_pos_absolute:].strip()
        
        # Validação: deve ter expressões decisórias próximas
        has_termos_decide = re.search(r'Termos em que se decide', decisao_text[:500], re.IGNORECASE)
        has_pelo_exposto = re.search(PELO_EXPOSTO_PATTERN, decisao_text[:500], re.IGNORECASE)
        has_verbos = re.search(VERBOS_DECISORIOS, decisao_text[:1000], re.IGNORECASE)
        
        if (has_termos_decide or has_pelo_exposto or has_verbos) and 50 < len(decisao_text) < 50000:
            resultado["decisao_extraida_do_texto_integral"] = decisao_text
            resultado["texto_integral_sem_decisao"] = texto_integral_completo[:start_pos_absolute].strip()
            resultado["metadata_decisao"] = {
                "metodo_extracao": "numeracao_romana_isolada",
                "confianca": "media",
                "palavra_chave_encontrada": last_match.group(1),
                "requer_verificacao_manual": False,
                "motivo_verificacao": None,
                "posicao_documento": f"{int((start_pos_absolute/texto_length)*100)}%"
            }
            return resultado
    
    # ========================================================================
    # CAMADA 5A-BIS: NUMERAÇÃO ALFABÉTICA ISOLADA - última ocorrência (A-F)
    # ========================================================================
    # Estratégia: alguns juízes usam APENAS letras sem nomes de secções
    # Exemplo: "C" isolado, seguido de texto da decisão
    # Procuramos a ÚLTIMA ocorrência de letras altas (A-F) no final
    numeracao_alfa_matches = list(re.finditer(NUMERACAO_ALFABETICA_ISOLADA_PATTERN, texto_final, re.MULTILINE))
    
    if numeracao_alfa_matches:
        # Usar a ÚLTIMA ocorrência (mais provável ser a decisão)
        last_match = numeracao_alfa_matches[-1]
        start_pos_absolute = inicio_busca + last_match.start()
        decisao_text = texto_integral_completo[start_pos_absolute:].strip()
        
        # Validação: deve ter expressões decisórias próximas
        has_termos_decide = re.search(r'Termos em que se decide', decisao_text[:500], re.IGNORECASE)
        has_pelo_exposto = re.search(PELO_EXPOSTO_PATTERN, decisao_text[:500], re.IGNORECASE)
        has_verbos = re.search(VERBOS_DECISORIOS, decisao_text[:1000], re.IGNORECASE)
        
        if (has_termos_decide or has_pelo_exposto or has_verbos) and 50 < len(decisao_text) < 50000:
            resultado["decisao_extraida_do_texto_integral"] = decisao_text
            resultado["texto_integral_sem_decisao"] = texto_integral_completo[:start_pos_absolute].strip()
            resultado["metadata_decisao"] = {
                "metodo_extracao": "numeracao_alfabetica_isolada",
                "confianca": "media",
                "palavra_chave_encontrada": last_match.group(1),
                "requer_verificacao_manual": False,
                "motivo_verificacao": None,
                "posicao_documento": f"{int((start_pos_absolute/texto_length)*100)}%"
            }
            return resultado
    
    # ========================================================================
    # CAMADA 5B: NUMERAÇÃO DECIMAL ISOLADA - última ocorrência (3., 4., 5., etc.)
    # NOTA: ÚLTIMA prioridade para numeração isolada - evita capturar
    # listas de factos probatórios como "82 - O Arguido agiu..."
    # ========================================================================
    # Estratégia: alguns juízes usam numeração decimal (1. Relatório, 2. Fundamentação, 3. Decisão)
    # Procuramos a ÚLTIMA numeração decimal alta (>=3) no final
    numeracao_decimal_matches = list(re.finditer(NUMERACAO_DECIMAL_PATTERN, texto_final, re.MULTILINE))
    
    if numeracao_decimal_matches:
        # Usar a ÚLTIMA ocorrência (mais provável ser a decisão)
        last_match = numeracao_decimal_matches[-1]
        start_pos_absolute = inicio_busca + last_match.start()
        decisao_text = texto_integral_completo[start_pos_absolute:].strip()
        
        # Validação FORTE: deve ter expressões decisórias próximas
        has_termos_decide = re.search(r'Termos em que se decide', decisao_text[:500], re.IGNORECASE)
        has_pelo_exposto = re.search(PELO_EXPOSTO_PATTERN, decisao_text[:500], re.IGNORECASE)
        has_verbos = re.search(VERBOS_DECISORIOS, decisao_text[:1000], re.IGNORECASE)
        
        if (has_termos_decide or has_pelo_exposto or has_verbos) and 50 < len(decisao_text) < 50000:
            resultado["decisao_extraida_do_texto_integral"] = decisao_text
            resultado["texto_integral_sem_decisao"] = texto_integral_completo[:start_pos_absolute].strip()
            resultado["metadata_decisao"] = {
                "metodo_extracao": "numeracao_decimal",
                "confianca": "media-baixa",
                "palavra_chave_encontrada": f"Numeral {last_match.group(1).strip()}",
                "requer_verificacao_manual": True,
                "motivo_verificacao": "Numeração decimal isolada - pode ser lista de factos - validar",
                "posicao_documento": f"{int((start_pos_absolute/texto_length)*100)}%"
            }
            return resultado
    
    # ========================================================================
    # CAMADA 6: THRESHOLD ADAPTATIVO - tentar com thresholds mais baixos
    # ========================================================================
    # Se chegamos aqui, não encontrámos nada com 70% threshold
    # Vamos tentar 60% (últimos 40%) e depois 50% (últimos 50%)
    # Isto ajuda em documentos com referências/notas extensas no final
    
    if inicio_busca == int(texto_length * 0.70):  # Se estamos no threshold inicial
        for new_threshold in [0.60, 0.50]:
            new_inicio_busca = int(texto_length * new_threshold)
            new_texto_final = texto_integral_completo[new_inicio_busca:]
            
            # Tentar apenas padrões mais confiáveis (DISPOSITIVO e DECISÃO numerados)
            dispositivo_retry = list(re.finditer(DISPOSITIVO_PATTERN, new_texto_final, re.IGNORECASE | re.MULTILINE))
            if dispositivo_retry:
                first_disp = dispositivo_retry[0]
                start_pos = new_inicio_busca + first_disp.start()
                dec_text = texto_integral_completo[start_pos:].strip()
                if re.search(VERBOS_DECISORIOS, dec_text[:1000], re.IGNORECASE) and 50 < len(dec_text) < 50000:
                    resultado["decisao_extraida_do_texto_integral"] = dec_text
                    resultado["texto_integral_sem_decisao"] = texto_integral_completo[:start_pos].strip()
                    resultado["metadata_decisao"] = {
                        "metodo_extracao": "dispositivo_numerado_threshold_adaptativo",
                        "confianca": "media-alta",
                        "palavra_chave_encontrada": first_disp.group(1).strip(),
                        "requer_verificacao_manual": False,
                        "motivo_verificacao": f"Encontrado com threshold {int(new_threshold*100)}% (documento longo)"
                    }
                    return resultado
            
            decisao_retry = list(re.finditer(DECISAO_PATTERN, new_texto_final, re.IGNORECASE | re.MULTILINE))
            if decisao_retry:
                first_dec = decisao_retry[0]
                start_pos = new_inicio_busca + first_dec.start()
                dec_text = texto_integral_completo[start_pos:].strip()
                if re.search(VERBOS_DECISORIOS, dec_text[:1000], re.IGNORECASE) and 50 < len(dec_text) < 50000:
                    resultado["decisao_extraida_do_texto_integral"] = dec_text
                    resultado["texto_integral_sem_decisao"] = texto_integral_completo[:start_pos].strip()
                    resultado["metadata_decisao"] = {
                        "metodo_extracao": "decisao_numerada_threshold_adaptativo",
                        "confianca": "media",
                        "palavra_chave_encontrada": first_dec.group(1).strip(),
                        "requer_verificacao_manual": False,
                        "motivo_verificacao": f"Encontrado com threshold {int(new_threshold*100)}% (documento longo)"
                    }
                    return resultado
    
    # ========================================================================
    # CAMADA 7: EDGE CASE FINAL - Nenhum padrão encontrado mesmo com threshold adaptativo
    # ========================================================================
    resultado["decisao_extraida_do_texto_integral"] = None
    resultado["texto_integral_sem_decisao"] = texto_integral_completo
    resultado["metadata_decisao"] = {
        "metodo_extracao": "none",
        "confianca": "baixa",
        "palavra_chave_encontrada": None,
        "requer_verificacao_manual": True,
        "motivo_verificacao": "Nenhum padrão de decisão encontrado mesmo com threshold adaptativo - estrutura atípica"
    }
    
    return resultado


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
        r'data\s+da\s+senten[cç]a',
        r'n[úu]mero\s+do\s+documento',
        r'n[ºo°]\s+do\s+documento',  # "Nº do Documento"
        r'processo:',
        r'relator:',
        r'tribunal:',
        r'secção:',
        r'meio\s+processual:',
        r'decisão:',
        r'votação:',
        r'texto\s+integral',
        r'julgado\s+de\s+paz',
        r'^[0-9\s\-/]+$',  # apenas números e traços
        r'^\s*\d+\s*$'  # apenas números
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

# def extract_sumario_advanced(soup):
#     """
#     Extrai o sumário usando múltiplas estratégias para maior robustez.
#     Retorna o sumário limpo ou None se não encontrado.
#     """
#     sumario_text = None
    
#     # Estratégia 1: Buscar por tag <b> ou <strong> com "Sumário"
#     sumario_tags = soup.find_all(['b', 'strong'], string=re.compile(r'sum[aá]rio', re.IGNORECASE))
    
#     for tag in sumario_tags:
#         content_parts = []
        
#         # Buscar conteúdo nos siblings seguintes
#         for sibling in tag.find_next_siblings():
#             if not sibling:
#                 continue
                
#             # Parar se encontrar outros títulos principais
#             if sibling.name in ['b', 'strong'] and re.search(
#                 r'(decis[aã]o|acordam|relatório|voto|fundamentos)', 
#                 sibling.get_text(), 
#                 re.IGNORECASE
#             ):
#                 break
                
#             # Extrair texto do sibling
#             text = sibling.get_text(strip=True) if hasattr(sibling, 'get_text') else str(sibling).strip()
#             if text and len(text) > 10:  # Ignorar textos muito curtos
#                 content_parts.append(text)
        
#         if content_parts:
#             sumario_text = "\n".join(content_parts).strip()
#             break
    
#     # Estratégia 2: Buscar por regex no texto completo se Estratégia 1 falhou
#     if not sumario_text:
#         full_text = soup.get_text(separator='\n', strip=True)
        
#         # Padrão específico: capturar tudo entre "Sumário:" e "Decisão Texto Integral/Parcial:"
#         sumario_match = re.search(
#             r'sum[aá]rio:?\s*\n?(.*?)(?=\n\s*(?:decis[aã]o\s+texto\s+(?:integral|parcial)|decis[aã]o|relatório|acordam|fundamentos|voto):|$)', 
#             full_text, 
#             re.IGNORECASE | re.DOTALL
#         )
        
#         if sumario_match:
#             sumario_text = sumario_match.group(1).strip()
    
#     # Estratégia 3: Buscar em estrutura de tabela (comum no DGSI)
#     if not sumario_text:
#         table_cells = soup.find_all(['td', 'th'])
#         for cell in table_cells:
#             cell_text = cell.get_text(strip=True)
#             if re.match(r'sum[aá]rio:?', cell_text, re.IGNORECASE):
#                 # Buscar célula seguinte ou próximo sibling
#                 next_cell = cell.find_next_sibling(['td', 'th'])
#                 if next_cell:
#                     sumario_text = next_cell.get_text(strip=True)
#                     break
    
#     # Limpeza final e validação
#     if sumario_text:
#         # Remover linhas muito curtas e limpeza geral
#         lines = [line.strip() for line in sumario_text.split('\n') if line.strip()]
#         clean_lines = [line for line in lines if len(line) > 5]  # Remover linhas muito curtas
        
#         # Filtrar linhas que são apenas pontos ou caracteres repetidos (ex: "………………")
#         clean_lines = [line for line in clean_lines if not re.match(r'^[.\s…]+$', line)]
        
#         # Remover linha "Acordam no Tribunal..." se for a primeira linha
#         if clean_lines and re.match(r'acordam\s+(?:no|os)', clean_lines[0], re.IGNORECASE):
#             clean_lines = clean_lines[1:]
        
#         if clean_lines:
#             sumario_text = "\n".join(clean_lines)
            
#             # Validação: sumário deve ter pelo menos 50 caracteres para ser válido
#             if len(sumario_text) >= 50:
#                 return sumario_text
    
#     return None

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
    Inclui extração avançada da decisão para prevenir data leakage.
    """
    metadata = {
        # Campos obrigatórios (identificação)
        "url": url,
        "tribunal": tribunal_nome,
        "tipo_direito": tipo_direito,
        "tipo_caso": tipo_caso,
        
        # Campos comuns (extraídos automaticamente)
        "n_processo": None,
        "juiz_relator": None,
        "data_acordao": None,
        "descritores": [],
        "votacao": None,
        "meio_processual": None,
        "decisao": None,  # Decisão do cabeçalho (ex: "CONFIRMADA", "NEGADO PROVIMENTO")
        "sumario": None,
        
        # Campos booleanos/específicos
        "texto_integral_disponivel": None,
        "texto_integral_completo": None,
        
        # NOVOS CAMPOS para prevenção de data leakage
        "decisao_extraida_do_texto_integral": None,
        "texto_integral_sem_decisao": None,
        "metadata_decisao": {},
        
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
    # metadata["sumario"] = extract_sumario_advanced(soup)

    # Extrair MÚLTIPLOS DESCRITORES (melhor parsing)
    descritores_match = re.search(r"Descritores:\s*([^\n]+(?:\n[^\n:]*)*)", text, re.IGNORECASE | re.MULTILINE)
    if descritores_match:
        descritores_text = descritores_match.group(1).strip()
        metadata["descritores"] = clean_descritores(descritores_text)
    else:
        metadata["descritores"] = []

    # Extrair Sumário usando função avançada
    # metadata["sumario"] = extract_sumario_advanced(soup)

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
    
    # NOVA FUNCIONALIDADE: Extrair decisão do texto integral para prevenção de data leakage
    if metadata["texto_integral_completo"]:
        decisao_resultado = extract_decisao_from_text(metadata["texto_integral_completo"])
        metadata["decisao_extraida_do_texto_integral"] = decisao_resultado["decisao_extraida_do_texto_integral"]
        metadata["texto_integral_sem_decisao"] = decisao_resultado["texto_integral_sem_decisao"]
        metadata["metadata_decisao"] = decisao_resultado["metadata_decisao"]

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
            output_filename = f"dgsi_tr_violencia_domestica_{tribunal_escolhido.lower()}_final_look.json"
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
