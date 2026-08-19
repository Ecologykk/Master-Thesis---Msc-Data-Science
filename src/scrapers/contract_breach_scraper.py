"""
Contract Breach Scraper for Portuguese Peace Courts (Julgados de Paz)

This module extracts and processes legal documents related to contract breach cases
from Portuguese Julgados de Paz (Justice of the Peace Courts).

Features:
- Extracts judgment metadata (process number, judge, date, descriptors, etc.)
- Separates final decision from judgment body to prevent data leakage in ML models
- Uses the same robust 15-layer cascade algorithm as domestic violence scraper


Author: Helton Mendonça

Transparency Note: This code was optimized and polished using AI assistance(mainly Claude 4.0 and 4.5). However, all the logic, structure and final review were done by the author.

Date: October 13th 2025
"""

import re
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler


class ContractBreachScraper:
    """
    Scraper for contract breach cases from Portuguese Peace Courts.

    Supports extraction from all Portuguese Julgados de Paz that publish
    decisions on dgsi.pt.
    """

    def __init__(self):
        """Initialize the scraper with peace court URLs."""
        self.court_urls = {
            "JP": "https://www.dgsi.pt/jpaz.nsf/Pesquisa+Descritor?OpenForm",  # JP - Julgados de Paz - Courts of Peace. Note: If you press this link it won't work, but trough the script it will. I don't know why.
            "CSM": "https://jurisprudencia.csm.org.pt/",  # CSM - Conselho Superior da Magistratura - Superior Council of the Magistracy
        }

    async def scrape_judgment(
        self, url, court_name="JP", case_type="INCUMPRIMENTO DE CONTRATOS"
    ):
        """
        Scrape a single judgment from Julgados de Paz.

        Args:
            url (str): URL of the judgment page
            court_name (str): Name of the court (JP for Julgados de Paz)
            case_type (str): Type of case (default: INCUMPRIMENTO DE CONTRATOS)

        Returns:
            dict: Complete metadata including separated decision text
        """
        try:
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(url=url)

                if not result.success:
                    return None

                soup = BeautifulSoup(result.html, "html.parser")
                metadata = self._extract_metadata(soup, url, court_name, case_type)

                return metadata

        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None

    def _extract_metadata(self, soup, url, court_name, case_type):
        """
        Extract all metadata from judgment HTML.

        Args:
            soup: BeautifulSoup parsed HTML
            url: Judgment URL
            court_name: Court identifier
            case_type: Type of legal case

        Returns:
            dict: Structured metadata with all fields
        """
        metadata = {
            "url": url,
            "tribunal": court_name,
            "tipo_direito": "CÍVEL",
            "tipo_caso": case_type,
            "n_processo": None,
            "juiz_relator": None,
            "data_acordao": None,
            "descritores": [],
            "votacao": None,
            "meio_processual": None,
            "decisao": None,
            "sumario": None,
            "texto_integral_disponivel": None,
            "texto_integral_completo": None,
            "decisao_extraida_do_texto_integral": None,
            "texto_integral_sem_decisao": None,
            "metadata_decisao": {},
            "campos_especificos": {},
        }

        text = soup.get_text(separator="\n", strip=True)

        # Extract basic fields - Peace Courts use different field names
        patterns = {
            "n_processo": r"Processo:\s*([^\n]+)",
            "juiz_relator": r"Relator:\s*([^\n]+)",
            "data_sentenca": r"Data\s+da\s+Senten[çc]a:\s*(\d{2}-\d{2}-\d{4})",  # JP only
            "data_acordao": r"Data\s+do\s+Ac[oó]rd[ãa]o:\s*(\d{2}-\d{2}-\d{4})",  # DGSI/CSM
            "decisao": r"Decis[ãa]o:\s*([^\n]+)",
            "texto_integral_disponivel": r"Texto Integral:\s*([SN])",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata[key] = match.group(1).strip()

        # Normalize date field: JP uses "Data da sentença", DGSI/CSM use "Data do Acordão"
        # Always store in "data_acordao" field for consistency
        if "data_sentenca" in metadata and metadata["data_sentenca"]:
            metadata["data_acordao"] = metadata.pop("data_sentenca")
        elif metadata["data_acordao"] is None:
            # Try to find any date pattern as fallback
            date_fallback = re.search(
                r"Data[^:]*:\s*(\d{2}-\d{2}-\d{4})", text, re.IGNORECASE
            )
            if date_fallback:
                metadata["data_acordao"] = date_fallback.group(1).strip()

        # Extract tribunal with locality (separate handling for proper formatting)
        tribunal_pattern = (
            r"Julgado\s+de\s+Paz\s+de\s*:?\s*([A-ZÀÁÂÃÇÉÊÍÓÔÕÚ][A-ZÀÁÂÃÇÉÊÍÓÔÕÚ\s\-]+)"
        )
        tribunal_match = re.search(tribunal_pattern, text, re.IGNORECASE)

        if tribunal_match:
            localidade = tribunal_match.group(1).strip()
            # Clean up extra whitespace and format as JP_LOCALITY
            localidade_clean = re.sub(r"\s+", "_", localidade.upper())
            metadata["tribunal"] = f"JP_{localidade_clean}"
        else:
            # Fallback: keep generic JP if locality not found
            metadata["tribunal"] = court_name  # Will be "JP"

        # Extract descriptors
        metadata["descritores"] = self._extract_descriptors(text)

        # Extract summary
        metadata["sumario"] = self._extract_summary(soup)

        # Extract full judgment text
        metadata["texto_integral_completo"] = self._extract_full_text(soup)

        # Extract and separate final decision (using same algorithm as domestic violence)
        if metadata["texto_integral_completo"]:
            decision_result = self._extract_decision_from_text(
                metadata["texto_integral_completo"]
            )
            metadata["decisao_extraida_do_texto_integral"] = decision_result[
                "decisao_extraida"
            ]
            metadata["texto_integral_sem_decisao"] = decision_result[
                "texto_sem_decisao"
            ]
            metadata["metadata_decisao"] = decision_result["metadata"]

        return metadata

    def _extract_descriptors(self, text):
        """Extract legal descriptors from text."""
        descriptors_match = re.search(
            r"Descritores:\s*([^\n]+(?:\n[^\n:]*)*)", text, re.IGNORECASE | re.MULTILINE
        )

        if not descriptors_match:
            return []

        descriptors_text = descriptors_match.group(1).strip()
        descriptors_list = re.split(r"[;,\n]+", descriptors_text)

        # Filter out invalid entries
        exclude_patterns = [
            r"data\s+da\s+senten[çc]a",
            r"processo:",
            r"ju[ií]z:",
            r"tribunal:",
            r"decisão:",
            r"área:",
            r"^[0-9\s\-/]+$",
        ]

        valid_descriptors = []
        for d in descriptors_list:
            d_clean = d.strip()
            if len(d_clean) > 2:
                if not any(
                    re.search(pattern, d_clean, re.IGNORECASE)
                    for pattern in exclude_patterns
                ):
                    valid_descriptors.append(d_clean)

        return valid_descriptors

    def _extract_summary(self, soup):
        """Extract judgment summary using multiple strategies."""
        # Strategy 1: Look for <b> or <strong> tags with "Sumário"
        summary_tags = soup.find_all(
            ["b", "strong"], string=re.compile(r"sum[aá]rio", re.IGNORECASE)
        )

        for tag in summary_tags:
            content_parts = []
            for sibling in tag.find_next_siblings():
                if sibling.name in ["b", "strong"]:
                    break
                text = (
                    sibling.get_text(strip=True)
                    if hasattr(sibling, "get_text")
                    else str(sibling).strip()
                )
                if text:
                    content_parts.append(text)

            if content_parts:
                summary_text = "\n".join(content_parts)
                if len(summary_text) > 20:
                    return summary_text

        # Strategy 2: Regex in full text
        full_text = soup.get_text(separator="\n", strip=True)
        summary_match = re.search(
            r"sum[aá]rio:?\s*\n?(.*?)(?=\n\s*(?:decis[aã]o\s+texto|relatório|acordam):|$)",
            full_text,
            re.IGNORECASE | re.DOTALL,
        )

        if summary_match:
            summary_text = summary_match.group(1).strip()
            if len(summary_text) > 20:
                return summary_text

        return None

    def _extract_full_text(self, soup):
        """Extract complete judgment text."""
        for element in soup(["script", "style", "nav", "header", "footer", "form"]):
            element.decompose()

        full_text = soup.get_text(separator="\n", strip=True)

        # Look for "Decisão Texto Integral:" or "Sentença:"
        decision_match = re.search(
            r"((?:decis[aã]o|senten[çc]a)\s+texto\s+integral:.*)",
            full_text,
            re.IGNORECASE | re.DOTALL,
        )

        if decision_match:
            return decision_match.group(1).strip()

        return full_text

    def _extract_decision_from_text(self, full_text):
        """
        Extract final decision from judgment text using progressive cascade algorithm.

        This algorithm is replicated from the domestic violence scraper to ensure
        consistent decision extraction across all case types. It prevents data leakage
        by separating the final decision section from the judgment reasoning.

        Strategy:
        - Focus on last 30% of document (where decisions typically appear)
        - Apply 15-layer cascade from most specific to most generic patterns
        - Validate with decision-related verbs and expressions
        - Use adaptive threshold (70% -> 60% -> 50%) for long documents

        Note: Again, this algorithm is not perfect and may require manual review for edge cases or atypical structures of the judgments.

        Args:
            full_text (str): Complete judgment text

        Returns:
            dict: Contains separated decision, remaining text, and metadata
        """
        if not full_text or len(full_text) < 100:
            return {
                "decisao_extraida": None,
                "texto_sem_decisao": full_text if full_text else "",
                "metadata": {
                    "extraction_method": "none",
                    "confidence": "low",
                    "keyword_found": None,
                    "requires_manual_review": True,
                    "review_reason": "Text too short or empty",
                },
            }

        # Start with 70% threshold (search in last 30%)
        text_length = len(full_text)
        search_start = int(text_length * 0.70)
        search_text = full_text[search_start:]

        # Define patterns - expanded judicial keywords
        DECISION_KEYWORDS = r"(?:DISPOSITIVO|Dispositivo|DECISÃO|Decisão|CONCLUSÃO|Conclusão|DELIBERAÇÃO|Deliberação|SENTENÇA|Sentença|VEREDICTO|Veredicto)"

        # Layer 1: Explicit section markers with roman numerals (III-X)
        DISPOSITIVO_ROMAN = rf"(?:^|\n)\s*((?:III|IV|V|VI|VII|VIII|IX|X)\s*[\s.\-–—:]*\s*{DECISION_KEYWORDS})[:.\s]*"
        DECISAO_ROMAN = rf"(?:^|\n)\s*((?:III|IV|V|VI|VII|VIII|IX|X)\s*[\s.\-–—:]*\s*{DECISION_KEYWORDS})[:.\s]*"

        # Layer 1-BIS: Explicit section markers with letters (A-F)
        DISPOSITIVO_ALPHA = (
            rf"(?:^|\n)\s*((?:[A-F])\s*[\s.\-–—:]*\s*{DECISION_KEYWORDS})[:.\s]*"
        )
        DECISAO_ALPHA = (
            rf"(?:^|\n)\s*((?:[A-F])\s*[\s.\-–—:]*\s*{DECISION_KEYWORDS})[:.\s]*"
        )

        # Layer 2: CAPS-only keywords
        DISPOSITIVO_CAPS = (
            r"(?:^|\n)\s*(DISPOSITIVO|DECISÃO|CONCLUSÃO|DELIBERAÇÃO)[:.\s]*"
        )
        DECISAO_CAPS = r"(?:^|\n)\s*(DECISÃO|DISPOSITIVO|CONCLUSÃO|DELIBERAÇÃO)[:.\s]*"

        # Layer 2.5: Numbered sections with names
        DECIMAL_WITH_NAME = (
            rf"(?:^|\n)\s*([3-9]|[1-9][0-9])\s*[\s.\-–—:]*\s*{DECISION_KEYWORDS}"
        )
        ALPHA_WITH_NAME = rf"(?:^|\n)\s*([A-F])\s*[\s.\-–—:]*\s*{DECISION_KEYWORDS}"

        # Layer 3-4: Concluding expressions (HIGH PRIORITY to prevent false positives)
        CONCLUDING_EXPR = r"(Pelo|Face ao|Nestes|Nos)\s+(?:exposto|termos|que precede|quanto|termos\s+expostos|termos\s+em\s+que\s+(?:se\s+)?(?:decide|acordam))s?[,:]"
        ACORDAM_EXPR = r"(Acordam|Acordaram|acorda-se)\s+(?:os\s+[Jj]u[ií]zes|em\s+conferência|no\s+Tribunal|em\s+(?:negar|conceder|julgar|manter|revogar))?"

        # Layer 5: Isolated numbering (LOW PRIORITY to avoid false positives)
        ROMAN_ISOLATED = (
            r"(?:^|\n)\s*((?:III|IV|V|VI|VII|VIII|IX|X))[\s.\-–—:]*(?=\n|$)"
        )
        ALPHA_ISOLATED = r"(?:^|\n)\s*([A-F])[\s.\-–—:]*(?=\n|$)"
        DECIMAL_ISOLATED = (
            r"(?:^|\n)\s*([3-9]|[1-9][0-9])\s*[\s.\-–—:]*\s*(?:\n|[A-ZÀ-Ú])"
        )

        # Decision verbs for validation
        DECISION_VERBS = r"\b(julga-se|condena-se|absolve-se|confirma-se|anula-se|revoga-se|indefere-se|mantém-se|decide-se|determina-se|nega-se|negar|negado\s+provimento|dá-se|concede-se|procede|improcede|procedente|improcedente|manter|confirmar|revogar)\b"

        result = {
            "decisao_extraida": None,
            "texto_sem_decisao": full_text,
            "metadata": {
                "extraction_method": None,
                "confidence": None,
                "keyword_found": None,
                "requires_manual_review": False,
                "review_reason": None,
            },
        }

        # Try extraction with current threshold
        extracted = self._try_extract_with_patterns(
            full_text,
            search_text,
            search_start,
            text_length,
            DISPOSITIVO_ROMAN,
            DECISAO_ROMAN,
            DISPOSITIVO_ALPHA,
            DECISAO_ALPHA,
            DISPOSITIVO_CAPS,
            DECISAO_CAPS,
            DECIMAL_WITH_NAME,
            ALPHA_WITH_NAME,
            CONCLUDING_EXPR,
            ACORDAM_EXPR,
            ROMAN_ISOLATED,
            ALPHA_ISOLATED,
            DECIMAL_ISOLATED,
            DECISION_VERBS,
        )

        if extracted:
            return extracted

        # Adaptive threshold: try 60% and 50% if nothing found
        for threshold in [0.60, 0.50]:
            search_start = int(text_length * threshold)
            search_text = full_text[search_start:]

            extracted = self._try_extract_with_patterns(
                full_text,
                search_text,
                search_start,
                text_length,
                DISPOSITIVO_ROMAN,
                DECISAO_ROMAN,
                DISPOSITIVO_ALPHA,
                DECISAO_ALPHA,
                DISPOSITIVO_CAPS,
                DECISAO_CAPS,
                DECIMAL_WITH_NAME,
                ALPHA_WITH_NAME,
                CONCLUDING_EXPR,
                ACORDAM_EXPR,
                ROMAN_ISOLATED,
                ALPHA_ISOLATED,
                DECIMAL_ISOLATED,
                DECISION_VERBS,
            )

            if extracted:
                return extracted

        # No pattern found even with adaptive threshold
        result["metadata"] = {
            "extraction_method": "none",
            "confidence": "low",
            "keyword_found": None,
            "requires_manual_review": True,
            "review_reason": "No decision pattern found with adaptive threshold - atypical structure",
        }

        return result

    def _try_extract_with_patterns(
        self,
        full_text,
        search_text,
        search_start,
        text_length,
        disp_roman,
        dec_roman,
        disp_alpha,
        dec_alpha,
        disp_caps,
        dec_caps,
        decimal_name,
        alpha_name,
        concluding,
        acordam,
        roman_iso,
        alpha_iso,
        decimal_iso,
        verbs,
    ):
        """
        Try to extract decision using all patterns in priority order.
        Returns extracted result if successful, None otherwise.
        """

        # Layer 1A: DISPOSITIVO with roman numerals
        matches = list(
            re.finditer(disp_roman, search_text, re.IGNORECASE | re.MULTILINE)
        )
        if matches:
            return self._validate_and_extract(
                full_text, search_start, matches[0], verbs, "dispositivo_roman", "high"
            )

        # Layer 1A-BIS: DISPOSITIVO with letters
        matches = list(
            re.finditer(disp_alpha, search_text, re.IGNORECASE | re.MULTILINE)
        )
        if matches:
            return self._validate_and_extract(
                full_text, search_start, matches[0], verbs, "dispositivo_alpha", "high"
            )

        # Layer 1B: DISPOSITIVO CAPS
        matches = list(re.finditer(disp_caps, search_text, re.MULTILINE))
        if matches:
            return self._validate_and_extract(
                full_text, search_start, matches[0], verbs, "dispositivo_caps", "high"
            )

        # Layer 2A: DECISÃO with roman numerals
        matches = list(
            re.finditer(dec_roman, search_text, re.IGNORECASE | re.MULTILINE)
        )
        if matches:
            return self._validate_and_extract(
                full_text,
                search_start,
                matches[0],
                verbs,
                "decisao_roman",
                "medium-high",
            )

        # Layer 2A-BIS: DECISÃO with letters
        matches = list(
            re.finditer(dec_alpha, search_text, re.IGNORECASE | re.MULTILINE)
        )
        if matches:
            return self._validate_and_extract(
                full_text,
                search_start,
                matches[0],
                verbs,
                "decisao_alpha",
                "medium-high",
            )

        # Layer 2B: DECISÃO CAPS
        matches = list(re.finditer(dec_caps, search_text, re.MULTILINE))
        if matches:
            return self._validate_and_extract(
                full_text,
                search_start,
                matches[0],
                verbs,
                "decisao_caps",
                "medium-high",
            )

        # Layer 2.5: Decimal with name (last occurrence)
        matches = list(
            re.finditer(decimal_name, search_text, re.IGNORECASE | re.MULTILINE)
        )
        if matches:
            return self._validate_and_extract(
                full_text, search_start, matches[-1], verbs, "decimal_with_name", "high"
            )

        # Layer 2.5-BIS: Alpha with name (last occurrence)
        matches = list(
            re.finditer(alpha_name, search_text, re.IGNORECASE | re.MULTILINE)
        )
        if matches:
            return self._validate_and_extract(
                full_text, search_start, matches[-1], verbs, "alpha_with_name", "high"
            )

        # Layer 3: Concluding expressions (HIGH PRIORITY - moved up)
        matches = list(re.finditer(concluding, search_text, re.IGNORECASE))
        if matches:
            result = self._validate_and_extract(
                full_text,
                search_start,
                matches[0],
                verbs,
                "concluding_expression",
                "medium-high",
                require_verbs=True,
            )
            if result:
                return result

        # Layer 4: Acordam expressions
        matches = list(re.finditer(acordam, search_text, re.IGNORECASE))
        if matches:
            return self._validate_and_extract(
                full_text, search_start, matches[0], verbs, "acordam", "medium"
            )

        # Layer 5A: Roman isolated (LOW PRIORITY - moved down)
        matches = list(re.finditer(roman_iso, search_text, re.MULTILINE))
        if matches:
            result = self._validate_and_extract(
                full_text,
                search_start,
                matches[-1],
                verbs,
                "roman_isolated",
                "medium",
                require_verbs=True,
            )
            if result:
                return result

        # Layer 5A-BIS: Alpha isolated
        matches = list(re.finditer(alpha_iso, search_text, re.MULTILINE))
        if matches:
            result = self._validate_and_extract(
                full_text,
                search_start,
                matches[-1],
                verbs,
                "alpha_isolated",
                "medium",
                require_verbs=True,
            )
            if result:
                return result

        # Layer 5B: Decimal isolated (LOWEST PRIORITY - flag for review)
        matches = list(re.finditer(decimal_iso, search_text, re.MULTILINE))
        if matches:
            result = self._validate_and_extract(
                full_text,
                search_start,
                matches[-1],
                verbs,
                "decimal_isolated",
                "medium-low",
                require_verbs=True,
            )
            if result:
                result["metadata"]["requires_manual_review"] = True
                result["metadata"][
                    "review_reason"
                ] = "Isolated decimal numbering - may be fact list - please validate"
            return result

        return None

    def _validate_and_extract(
        self,
        full_text,
        search_start,
        match,
        verb_pattern,
        method_name,
        confidence,
        require_verbs=False,
    ):
        """
        Validate and extract decision text from a pattern match.

        Args:
            full_text: Complete judgment text
            search_start: Position where search began
            match: Regex match object
            verb_pattern: Pattern to find decision verbs
            method_name: Name of extraction method
            confidence: Confidence level (high/medium-high/medium/medium-low/low)
            require_verbs: Whether decision verbs are required for validation

        Returns:
            dict with extracted decision or None if validation fails
        """
        start_pos = search_start + match.start()
        decision_text = full_text[start_pos:].strip()

        # Check for decision verbs
        has_verbs = re.search(verb_pattern, decision_text[:1000], re.IGNORECASE)

        # Validate length and verb presence
        if require_verbs and not has_verbs:
            return None

        if not (50 < len(decision_text) < 50000):
            return None

        return {
            "decisao_extraida": decision_text,
            "texto_sem_decisao": full_text[:start_pos].strip(),
            "metadata": {
                "extraction_method": method_name,
                "confidence": confidence,
                "keyword_found": match.group(0).strip()[:50],
                "requires_manual_review": False,
                "review_reason": None,
                "document_position": f"{int((start_pos/len(full_text))*100)}%",
            },
        }
