## Domestic violence/Contract Breach scraper script

### Dependencies


| Tool         | Docs Link                                              | Version | Motive                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------ | ------------------------------------------------------ | ------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python       | https://www.python.org/downloads/release/python-31210/ | 3.12.10 | Easy to work for this task, has all necessary tools.                                                                                                                                                                                                                                                                                                                                                                  |
| BeutifulSoup | https://www.crummy.com/software/BeautifulSoup/         | 4.13.5  | Usef for HTML parsing and Document Object Model Trasversal (meaning efficiently find and manipulate HTML elements)                                                                                                                                                                                                                                                                                                    |
| Crawl4AI     | https://docs.crawl4ai.com/                             | 0.7.4   | Modern Alternative that lets scrape pages asyncronously and extract the text in a markdown format, making it easier for ML/DL ingestion. It was used to obtain the url from the pages yet it was not fully used to its capabilites as it was not deemed worth it. "requests" would of have same performance for this case, but would not allow further performance upgrades in the future for massive volume scraping |


### Code explaining
I will be using the Domestic violence script as example here, but everything is similar for the contract breach cases. 
#### 1. Importing



```python

import re

from bs4 import BeautifulSoup

from crawl4ai import AsyncWebCrawler

```

- **re** - Regex (Regular expression library), necessary for textual pattern extraction. 
- **asyncio** - Again this one was meant more for the concurrent approach, but kept it either way, it did not break the code. Also it was necessary to run craw4ai
- **datetime** - only really used to id files
- **beutiful soup** - the html parser that enables the metadata extraction and full judgment decision extraction.
- **crawl4ai** - this was first used as the concurrent scraper it basically used internally bs4 to parse the html and extract the text. Altough in our case in the more simple version i just used it simply to obtain the urls present in the webpage. In the sequential approach i took, libraries like "request" would of been just as good. However, craw4ai concurrent abilities allow for future re-work of the script.

#### 2. Per document scraping and pattern extraction Class

```python
class DomesticViolenceScraper:

```

##### 2.1 Initialization
This will be the class defining all actions that the specific domestic violence case scraping will have.

```python
class DomesticViolenceScraper:
 def __init__(self):

        """Initialize the scraper with court URLs."""

        self.court_urls = {

            "TRE": "https://www.dgsi.pt/jtre.nsf/Pesquisa+Descritor?OpenForm",

            "TRL": "https://www.dgsi.pt/jtrl.nsf/Pesquisa+Descritor?OpenForm",

            "TRC": "https://www.dgsi.pt/jtrc.nsf/Pesquisa+Descritor?OpenForm",

            "TRG": "https://www.dgsi.pt/jtrg.nsf/Pesquisa+Descritor?OpenForm",

            "TRP": "https://www.dgsi.pt/jtrp.nsf/Pesquisa+Descritor?OpenForm"

        }
        
```

The initialization method above makes a dictionary with all search pages for each specific court we will extract documents from.

##### 2.2  Url scrping logic
Next we have:

```python
async def scrape_judgment(self, url, court_name="Unknown", case_type="VIOLÊNCIA DOMÉSTICA"):

        """

        Scrape a single judgment from DGSI.

        Args:

            url (str): URL of the judgment page

            court_name (str): Name of the court (TRE, TRL, TRC, TRG, TRP)

            case_type (str): Type of case (default: VIOLÊNCIA DOMÉSTICA)

        Returns:

            dict: Complete metadata including separated decision text

        """

        try:

            async with AsyncWebCrawler(verbose=False) as crawler:

                result = await crawler.arun(url=url)

                if not result.success:

                    return None

                soup = BeautifulSoup(result.html, 'html.parser')

                metadata = self._extract_metadata(soup, url, court_name, case_type)

                return metadata

        except Exception as e:

            print(f"Error scraping {url}: {str(e)}")

            return None
```


This method uses at a first stage Crawl4ai, as html requester, by passing the specific case url, but keeps going in case it fails, this is to reduce bottlenceks in case document scraping fails, which was rare. Nonetheless is a nice habit to keep:

```python
async with AsyncWebCrawler(verbose=False) as crawler:

                result = await crawler.arun(url=url)

                if not result.success:

                    return None
```

Then it used beutifulsoup to read the html and extract the metadata, due to the DOM tranversal habilities it has. (more on this here: https://medium.com/@mohanapriyanpriyan4/dom-traversal-165061ab7254):

```python
soup = BeautifulSoup(result.html, 'html.parser')

                metadata = self._extract_metadata(soup, url, court_name, case_type)

                return metadata
```

Try/Exception will be common pattern, to validate scenarios and making the script more robust to errors, while making it easier to debug.

##### 2.2 Metadata extraction from html

We follow with the the method responsible for taking out of the html the metadata of a case, which came at the head of an document:

```python

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

            "tipo_direito": "PENAL",

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

            "campos_especificos": {}

        }

        text = soup.get_text(separator='\n', strip=True)

        # Extract basic fields

        patterns = {

            "n_processo": r"Processo:\s*([^\n]+)",

            "juiz_relator": r"Relator:\s*([^\n]+)",

            "data_acordao": r"Data do Acordão:\s*(\d{2}-\d{2}-\d{4})",

            "votacao": r"Votação:\s*([^\n]+)",

            "meio_processual": r"Meio Processual:\s*([^\n]+)",

            "decisao": r"Decisão:\s*([^\n]+)",

            "texto_integral_disponivel": r"Texto Integral:\s*([SN])",

        }

        for key, pattern in patterns.items():

            match = re.search(pattern, text, re.IGNORECASE)

            if match:

                metadata[key] = match.group(1).strip()

        # Extract descriptors

        metadata["descritores"] = self._extract_descriptors(text)

        # Extract summary

        metadata["sumario"] = self._extract_summary(soup)

        # Extract full judgment text

        metadata["texto_integral_completo"] = self._extract_full_text(soup)

        # Extract and separate final decision

        if metadata["texto_integral_completo"]:

            decision_result = self._extract_decision_from_text(metadata["texto_integral_completo"])

            metadata["decisao_extraida_do_texto_integral"] = decision_result["decisao_extraida"]

            metadata["texto_integral_sem_decisao"] = decision_result["texto_sem_decisao"]

            metadata["metadata_decisao"] = decision_result["metadata"]

        return metadata
```

The metadata that is extracted is quite self explanatory in their names. Note that most of this metadata is not strictly necessary for the prediction task at hands, but could be interesting/useful for future analysis. Some are raw extracted, others(namely the decision part extracted from the full text and some validation flags) were done trough algorithms.

- **url** - the url of the case, important for future referencing, when using llms for example.
- **tribunal**  - the court where the document was extractred from.
- **tipo_direito** - type of doctrine of the case. In our situation we only had penal cases (domestic violence) and civil cases (contract breaches). In a future, larger scale research this could be handy in a greater variety of cases.
- **tipo_caso** -  type of case (e.g Domestic Violence)
- **n_processo** - case unique id identifier, given by the actual court admin
- **juiz_relator**  -  Main judge, tasked with the writing of the final report. Could be interesting finding patterns per judge in the EDA, in the future.
- **data_acordão** - date of the case, important not only for information purposes but also for filtering if necessary.
- **descritores** - these are the list of keywords given by the judges, for the particular case. In general they can be commonly used by many, but some keywords might be more specific for some judges. Ideally we would want a normalized and unified list judges should use from. It seems that is not the case. This metadata is quite importante to get a sense or pattern in the cases, beyond the common domestic violence thread between them.
- **votacao** - if the voting was unanimous or not. Not really useful for my work.
- **meio_processual** - what type of case is in analysis (e.g if first instance, appeal , etc)
- **decisao** - short sentence/word of the final decision made by the judges. Most cases have this metadata. Altough some normalization will be necessary to enable the classification task. For those that do not, then the extracted decision will be crucial, another reason for extracting it.
- **sumario** - summary of the decision. Now this one is controversial because altough it says summary, most of the time is something vague, that even for humans is hard to understand what it actually the case is about and the analysis and judgments made. At first thought it would be very useful, particularly in the llm phase, but now i think will be better ditching it.
- **text_integral_disponivel** - if the full text is avaiable. This metadata is native for the documents, but not very useful as most documents have it. I imagine this was more useful back in the days when not all documents were avaiable in full format.
- **texto_integral_completo** - full case text from report to decision. This could be useful to extract *a posteriori* the final decision from those cases the regex did not work.
- **decisao_extraida_do_texto_integral** - Here we separate the final decision from the rest of the judgment text (report and justification). This is done to prevent data leakage in the ML modelling where the final decision could bias the models into the right model. By hiding it, the models must rely more on the report (context of the case) and the justification (judge reasoning of the case).
- **texto_integral_sem_decisao** - the case text without the final decision. This will be the main text used to actually get the textual features of each case.
- **metadata_decisao** - these were just some engineered metadata to understand why a certain extraction was failing or not, its not mandatory.
- **campos_especificos** - was meant to capture unique fields in the head of the document but its not mandatory for this research.


Then we extract the metadata, using specific methods for the descriptors, the summary which was not getting fully capture at first and the full text:



```python
for key, pattern in patterns.items():

            match = re.search(pattern, text, re.IGNORECASE)

            if match:

                metadata[key] = match.group(1).strip()

        # Extract descriptors

        metadata["descritores"] = self._extract_descriptors(text)

        # Extract summary

        metadata["sumario"] = self._extract_summary(soup)

        # Extract full judgment text

        metadata["texto_integral_completo"] = self._extract_full_text(soup)

        # Extract and separate final decision

        if metadata["texto_integral_completo"]:

            decision_result = self._extract_decision_from_text(metadata["texto_integral_completo"])

            metadata["decisao_extraida_do_texto_integral"] = decision_result["decisao_extraida"]

            metadata["texto_integral_sem_decisao"] = decision_result["texto_sem_decisao"]

            metadata["metadata_decisao"] = decision_result["metadata"]

        return metadata
```

To adress the descriptors, basic string manipulation and regex was implemented on the parsed html:

```python
 def _extract_descriptors(self, text):

        """Extract legal descriptors from text."""

        descriptors_match = re.search(

            r"Descritores:\s*([^\n]+(?:\n[^\n:]*)*)",

            text,

            re.IGNORECASE | re.MULTILINE

        )

        if not descriptors_match:

            return []

        descriptors_text = descriptors_match.group(1).strip()

        descriptors_list = re.split(r'[;,\n]+', descriptors_text)

        # Filter out invalid entries

        exclude_patterns = [

            r'data\s+do\s+acord[aã]o', r'processo:', r'relator:',

            r'tribunal:', r'decisão:', r'votação:', r'^[0-9\s\-/]+$'

        ]

        valid_descriptors = []

        for d in descriptors_list:

            d_clean = d.strip()

            if len(d_clean) > 2:

                if not any(re.search(pattern, d_clean, re.IGNORECASE) for pattern in exclude_patterns):

                    valid_descriptors.append(d_clean)

        return valid_descriptors

```

For the summary extraction, there were 2 main strategies:

One that would look for the flags  b or strong in the html, next to sumário. This was because pure regex was being slightly unreliable, which i dont exactly understand why.
Either way the summary will not be used for this research purposes, as they are usually vague and not really descriptive enough to understand the judgment, even for a human.

```python
 def _extract_summary(self, soup):

        """Extract judgment summary using multiple strategies."""

        # Strategy 1: Look for <b> or <strong> tags with "Sumário"

        summary_tags = soup.find_all(['b', 'strong'], string=re.compile(r'sum[aá]rio', re.IGNORECASE))

        for tag in summary_tags:

            content_parts = []

            for sibling in tag.find_next_siblings():

                if sibling.name in ['b', 'strong']:

                    break

                text = sibling.get_text(strip=True) if hasattr(sibling, 'get_text') else str(sibling).strip()

                if text:

                    content_parts.append(text)

            if content_parts:

                summary_text = '\n'.join(content_parts)

                if len(summary_text) > 20:

                    return summary_text
```

or going trough the full text, but only as last resort, and then again, summary will not be used.

```python
 # Strategy 2: Regex in full text

        full_text = soup.get_text(separator='\n', strip=True)

        summary_match = re.search(

            r'sum[aá]rio:?\s*\n?(.*?)(?=\n\s*(?:decis[aã]o\s+texto|relatório|acordam):|$)',

            full_text,

            re.IGNORECASE | re.DOTALL

        )

        if summary_match:

            summary_text = summary_match.group(1).strip()

            if len(summary_text) > 20:

                return summary_text

        return None
```

For the full text it was usually easier.
Just getting the right patterns in html and extracting after that:
```python
def _extract_full_text(self, soup):

        """Extract complete judgment text."""

        for element in soup(["script", "style", "nav", "header", "footer", "form"]):

            element.decompose()

        full_text = soup.get_text(separator='\n', strip=True)

        # Look for "Decisão Texto Integral:"

        decision_match = re.search(

            r'(decis[aã]o\s+texto\s+integral:.*)',

            full_text,

            re.IGNORECASE | re.DOTALL

        )

        if decision_match:

            return decision_match.group(1).strip()

        return full_text
```

##### 2.3 Final decision extraction from full text

```python
def _extract_decision_from_text(self, full_text):

        """

        Extract final decision from judgment text using progressive cascade algorithm.

        This is the core algorithm that separates the final decision (dispositivo)

        from the rest of the judgment to prevent data leakage in ML models.

        Strategy:

        - Focus on last 30% of document (where decisions typically appear)

        - Apply 15-layer cascade from most specific to most generic patterns

        - Validate with decision-related verbs and expressions

        - Use adaptive threshold (70% -> 60% -> 50%) for long documents

        Note: This algorithm is not perfect and may require manual review for edge cases or atypical structures of the judgments.

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

                    "review_reason": "Text too short or empty"

                }

            }

        # Start with 70% threshold (search in last 30%)

        text_length = len(full_text)

        search_start = int(text_length * 0.70)

        search_text = full_text[search_start:]

        # Define patterns - expanded judicial keywords

        DECISION_KEYWORDS = r'(?:DISPOSITIVO|Dispositivo|DECISÃO|Decisão|CONCLUSÃO|Conclusão|DELIBERAÇÃO|Deliberação|SENTENÇA|Sentença|VEREDICTO|Veredicto)'

        # Layer 1: Explicit section markers with roman numerals (III-X)

        DISPOSITIVO_ROMAN = rf'(?:^|\n)\s*((?:III|IV|V|VI|VII|VIII|IX|X)\s*[\s.\-–—:]*\s*{DECISION_KEYWORDS})[:.\s]*'

        DECISAO_ROMAN = rf'(?:^|\n)\s*((?:III|IV|V|VI|VII|VIII|IX|X)\s*[\s.\-–—:]*\s*{DECISION_KEYWORDS})[:.\s]*'

        # Layer 1-BIS: Explicit section markers with letters (A-F)

        DISPOSITIVO_ALPHA = rf'(?:^|\n)\s*((?:[A-F])\s*[\s.\-–—:]*\s*{DECISION_KEYWORDS})[:.\s]*'

        DECISAO_ALPHA = rf'(?:^|\n)\s*((?:[A-F])\s*[\s.\-–—:]*\s*{DECISION_KEYWORDS})[:.\s]*'

        # Layer 2: CAPS-only keywords

        DISPOSITIVO_CAPS = r'(?:^|\n)\s*(DISPOSITIVO|DECISÃO|CONCLUSÃO|DELIBERAÇÃO)[:.\s]*'

        DECISAO_CAPS = r'(?:^|\n)\s*(DECISÃO|DISPOSITIVO|CONCLUSÃO|DELIBERAÇÃO)[:.\s]*'

        # Layer 2.5: Numbered sections with names

        DECIMAL_WITH_NAME = rf'(?:^|\n)\s*([3-9]|[1-9][0-9])\s*[\s.\-–—:]*\s*{DECISION_KEYWORDS}'

        ALPHA_WITH_NAME = rf'(?:^|\n)\s*([A-F])\s*[\s.\-–—:]*\s*{DECISION_KEYWORDS}'

        # Layer 3-4: Concluding expressions (moved UP in priority)

        CONCLUDING_EXPR = r'(Pelo|Face ao|Nestes|Nos)\s+(?:exposto|termos|que precede|quanto|termos\s+expostos|termos\s+em\s+que\s+(?:se\s+)?(?:decide|acordam))s?[,:]'

        ACORDAM_EXPR = r'(Acordam|Acordaram|acorda-se)\s+(?:os\s+[Jj]u[ií]zes|em\s+conferência|no\s+Tribunal|em\s+(?:negar|conceder|julgar|manter|revogar))?'

        # Layer 5: Isolated numbering (moved DOWN in priority to avoid false positives)

        ROMAN_ISOLATED = r'(?:^|\n)\s*((?:III|IV|V|VI|VII|VIII|IX|X))[\s.\-–—:]*(?=\n|$)'

        ALPHA_ISOLATED = r'(?:^|\n)\s*([A-F])[\s.\-–—:]*(?=\n|$)'

        DECIMAL_ISOLATED = r'(?:^|\n)\s*([3-9]|[1-9][0-9])\s*[\s.\-–—:]*\s*(?:\n|[A-ZÀ-Ú])'

        # Decision verbs for validation

        DECISION_VERBS = r'\b(julga-se|condena-se|absolve-se|confirma-se|anula-se|revoga-se|indefere-se|mantém-se|decide-se|determina-se|nega-se|negar|negado\s+provimento|dá-se|concede-se|procede|improcede|procedente|improcedente|manter|confirmar|revogar)\b'

        result = {

            "decisao_extraida": None,

            "texto_sem_decisao": full_text,

            "metadata": {

                "extraction_method": None,

                "confidence": None,

                "keyword_found": None,

                "requires_manual_review": False,

                "review_reason": None

            }

        }

        # Try extraction with current threshold

        extracted = self._try_extract_with_patterns(

            full_text, search_text, search_start, text_length,

            DISPOSITIVO_ROMAN, DECISAO_ROMAN, DISPOSITIVO_ALPHA, DECISAO_ALPHA,

            DISPOSITIVO_CAPS, DECISAO_CAPS, DECIMAL_WITH_NAME, ALPHA_WITH_NAME,

            CONCLUDING_EXPR, ACORDAM_EXPR, ROMAN_ISOLATED, ALPHA_ISOLATED,

            DECIMAL_ISOLATED, DECISION_VERBS

        )

        if extracted:

            return extracted

        # Adaptive threshold: try 60% and 50% if nothing found

        for threshold in [0.60, 0.50]:

            search_start = int(text_length * threshold)

            search_text = full_text[search_start:]

            extracted = self._try_extract_with_patterns(

                full_text, search_text, search_start, text_length,

                DISPOSITIVO_ROMAN, DECISAO_ROMAN, DISPOSITIVO_ALPHA, DECISAO_ALPHA,

                DISPOSITIVO_CAPS, DECISAO_CAPS, DECIMAL_WITH_NAME, ALPHA_WITH_NAME,

                CONCLUDING_EXPR, ACORDAM_EXPR, ROMAN_ISOLATED, ALPHA_ISOLATED,

                DECIMAL_ISOLATED, DECISION_VERBS

            )

            if extracted:

                return extracted

        # No pattern found even with adaptive threshold

        result["metadata"] = {

            "extraction_method": "none",

            "confidence": "low",

            "keyword_found": None,

            "requires_manual_review": True,

            "review_reason": "No decision pattern found with adaptive threshold - atypical structure"

        }

        return result
```

We can start with a guideline I imposed, after finding extremely small docs(very rare though).
Where we simply dont extract any decision because the text has less than 100 elements in it (words, caracathers)

```python

if not full_text or len(full_text) < 100:

            return {

                "decisao_extraida": None,

                "texto_sem_decisao": full_text if full_text else "",

                "metadata": {

                    "extraction_method": "none",

                    "confidence": "low",

                    "keyword_found": None,

                    "requires_manual_review": True,

                    "review_reason": "Text too short or empty"

                }

            }
```

Then, to reduce error and make it more reliable, only focused on the end of the document, because the final decision is always at the end of it.

```python
# Start with 70% threshold (search in last 30%)

        text_length = len(full_text)

        search_start = int(text_length * 0.70)

        search_text = full_text[search_start:]

```

Subsequentially, it was important to set some possible patterns, for the regex to work. These were done trough a lot of trial and error and manually reviewing why the wrongly extracted or not extracted decisions were happening. The human in the loop + ai assistance here was extremely useful and time saving, had i tried doing it all by myself or give all the ai to do it.

**Pattern 1:** words that signify a judgment decision. Usually DISPOSITIVO and DECISÃO are the most common, but too be more flexible I added other synonyms

```python 
# Define patterns - expanded judicial keywords

        DECISION_KEYWORDS = r'(?:DISPOSITIVO|Dispositivo|DECISÃO|Decisão|CONCLUSÃO|Conclusão|DELIBERAÇÃO|Deliberação|SENTENÇA|Sentença|VEREDICTO|Veredicto)'
```

**Pattern 2:** Usually main sections (Report, Justification, Decision and all subsections) are usually numbered with roman numerals or Alphabetic letters. [^2]They can also start with decimal numbers, but those can be less reliable by themselves because it can also mean numbering of just specific points judges might be adressing, like facts or evidence.

```python
 # Layer 1: Explicit section markers with roman numerals (III-X)

        DISPOSITIVO_ROMAN = rf'(?:^|\n)\s*((?:III|IV|V|VI|VII|VIII|IX|X)\s*[\s.\-–—:]*\s*{DECISION_KEYWORDS})[:.\s]*'

        DECISAO_ROMAN = rf'(?:^|\n)\s*((?:III|IV|V|VI|VII|VIII|IX|X)\s*[\s.\-–—:]*\s*{DECISION_KEYWORDS})[:.\s]*'

        # Layer 1-BIS: Explicit section markers with letters (A-F)

        DISPOSITIVO_ALPHA = rf'(?:^|\n)\s*((?:[A-F])\s*[\s.\-–—:]*\s*{DECISION_KEYWORDS})[:.\s]*'

        DECISAO_ALPHA = rf'(?:^|\n)\s*((?:[A-F])\s*[\s.\-–—:]*\s*{DECISION_KEYWORDS})[:.\s]*'

```

**Pattern 3:** Judges might also simply not number, but just name the sections. Usually in caps, to distinguish from other sections. 

```python

# Layer 2: CAPS-only keywords

        DISPOSITIVO_CAPS = r'(?:^|\n)\s*(DISPOSITIVO|DECISÃO|CONCLUSÃO|DELIBERAÇÃO)[:.\s]*'

        DECISAO_CAPS = r'(?:^|\n)\s*(DECISÃO|DISPOSITIVO|CONCLUSÃO|DELIBERAÇÃO)[:.\s]*'

        # Layer 2.5: Numbered sections with names

        DECIMAL_WITH_NAME = rf'(?:^|\n)\s*([3-9]|[1-9][0-9])\s*[\s.\-–—:]*\s*{DECISION_KEYWORDS}'

        ALPHA_WITH_NAME = rf'(?:^|\n)\s*([A-F])\s*[\s.\-–—:]*\s*{DECISION_KEYWORDS}'
```

**Pattern4:** we can use concluding expressions to extract the decision. Ideally this should be low in priority, due to their subjective and volatitle nature.

```python
 # Layer 3-4: Concluding expressions (moved UP in priority)

        CONCLUDING_EXPR = r'(Pelo|Face ao|Nestes|Nos)\s+(?:exposto|termos|que precede|quanto|termos\s+expostos|termos\s+em\s+que\s+(?:se\s+)?(?:decide|acordam))s?[,:]'

        ACORDAM_EXPR = r'(Acordam|Acordaram|acorda-se)\s+(?:os\s+[Jj]u[ií]zes|em\s+conferência|no\s+Tribunal|em\s+(?:negar|conceder|julgar|manter|revogar))?'
```

**Pattern 5**: Finally we can just rely on the alphanumeric numbering in isolated form. But this is low in priority to reduce false positives, because without context (other key words) these might catch just a enumeration in other sections. We also add some validation words to use further down to validate if actually it was a decision extracted or not. Also we n use them in the future to extract decisions for classification more explcitly.

```python

# Layer 5: Isolated numbering (moved DOWN in priority to avoid false positives)

        ROMAN_ISOLATED = r'(?:^|\n)\s*((?:III|IV|V|VI|VII|VIII|IX|X))[\s.\-–—:]*(?=\n|$)'

        ALPHA_ISOLATED = r'(?:^|\n)\s*([A-F])[\s.\-–—:]*(?=\n|$)'

        DECIMAL_ISOLATED = r'(?:^|\n)\s*([3-9]|[1-9][0-9])\s*[\s.\-–—:]*\s*(?:\n|[A-ZÀ-Ú])'

        # Decision verbs for validation

        DECISION_VERBS = r'\b(julga-se|condena-se|absolve-se|confirma-se|anula-se|revoga-se|indefere-se|mantém-se|decide-se|determina-se|nega-se|negar|negado\s+provimento|dá-se|concede-se|procede|improcede|procedente|improcedente|manter|confirmar|revogar)\b'
```


The final output of this method is now computed by applying the patterns and the final result is the decision extracted and the text without the final decision. the metadata field is just some flags to understand which patters failed or to manually review lower confidence cases (the ones that use the lower priority patterns).

Also the adaptive threshold happens here, where if we did not find the decision we use 60% and then ultimately 50%. This usually gets the decision. Resorted more for the longer documents. The shorter ones are not really interesting cause might be not information on them (usually extremely old cases are like these, where the means to digitilize docs were not as good)

```python

result = {

            "decisao_extraida": None,

            "texto_sem_decisao": full_text,

            "metadata": {

                "extraction_method": None,

                "confidence": None,

                "keyword_found": None,

                "requires_manual_review": False,

                "review_reason": None

            }

        }

        # Try extraction with current threshold

        extracted = self._try_extract_with_patterns(

            full_text, search_text, search_start, text_length,

            DISPOSITIVO_ROMAN, DECISAO_ROMAN, DISPOSITIVO_ALPHA, DECISAO_ALPHA,

            DISPOSITIVO_CAPS, DECISAO_CAPS, DECIMAL_WITH_NAME, ALPHA_WITH_NAME,

            CONCLUDING_EXPR, ACORDAM_EXPR, ROMAN_ISOLATED, ALPHA_ISOLATED,

            DECIMAL_ISOLATED, DECISION_VERBS

        )

        if extracted:

            return extracted

        # Adaptive threshold: try 60% and 50% if nothing found

        for threshold in [0.60, 0.50]:

            search_start = int(text_length * threshold)

            search_text = full_text[search_start:]

            extracted = self._try_extract_with_patterns(

                full_text, search_text, search_start, text_length,

                DISPOSITIVO_ROMAN, DECISAO_ROMAN, DISPOSITIVO_ALPHA, DECISAO_ALPHA,

                DISPOSITIVO_CAPS, DECISAO_CAPS, DECIMAL_WITH_NAME, ALPHA_WITH_NAME,

                CONCLUDING_EXPR, ACORDAM_EXPR, ROMAN_ISOLATED, ALPHA_ISOLATED,

                DECIMAL_ISOLATED, DECISION_VERBS

            )

            if extracted:

                return extracted

        # No pattern found even with adaptive threshold

        result["metadata"] = {

            "extraction_method": "none",

            "confidence": "low",

            "keyword_found": None,

            "requires_manual_review": True,

            "review_reason": "No decision pattern found with adaptive threshold - atypical structure"

        }

        return result
```


The auxiliary method _try_extract_with_patterns(), will actually impose in the patterns in order of importance with logic rules (if,else).

```python
 def _try_extract_with_patterns(self, full_text, search_text, search_start, text_length,

                                   disp_roman, dec_roman, disp_alpha, dec_alpha,

                                   disp_caps, dec_caps, decimal_name, alpha_name,

                                   concluding, acordam, roman_iso, alpha_iso, decimal_iso, verbs):

        """

        Try to extract decision using all patterns in priority order.

        Returns extracted result if successful, None otherwise.

        """

        # Layer 1A: DISPOSITIVO with roman numerals

        matches = list(re.finditer(disp_roman, search_text, re.IGNORECASE | re.MULTILINE))

        if matches:

            return self._validate_and_extract(full_text, search_start, matches[0], verbs,

                                              "dispositivo_roman", "high")

        # Layer 1A-BIS: DISPOSITIVO with letters

        matches = list(re.finditer(disp_alpha, search_text, re.IGNORECASE | re.MULTILINE))

        if matches:

            return self._validate_and_extract(full_text, search_start, matches[0], verbs,

                                              "dispositivo_alpha", "high")

        # Layer 1B: DISPOSITIVO CAPS

        matches = list(re.finditer(disp_caps, search_text, re.MULTILINE))

        if matches:

            return self._validate_and_extract(full_text, search_start, matches[0], verbs,

                                              "dispositivo_caps", "high")

        # Layer 2A: DECISÃO with roman numerals

        matches = list(re.finditer(dec_roman, search_text, re.IGNORECASE | re.MULTILINE))

        if matches:

            return self._validate_and_extract(full_text, search_start, matches[0], verbs,

                                              "decisao_roman", "medium-high")

        # Layer 2A-BIS: DECISÃO with letters

        matches = list(re.finditer(dec_alpha, search_text, re.IGNORECASE | re.MULTILINE))

        if matches:

            return self._validate_and_extract(full_text, search_start, matches[0], verbs,

                                              "decisao_alpha", "medium-high")

        # Layer 2B: DECISÃO CAPS

        matches = list(re.finditer(dec_caps, search_text, re.MULTILINE))

        if matches:

            return self._validate_and_extract(full_text, search_start, matches[0], verbs,

                                              "decisao_caps", "medium-high")

        # Layer 2.5: Decimal with name (last occurrence)

        matches = list(re.finditer(decimal_name, search_text, re.IGNORECASE | re.MULTILINE))

        if matches:

            return self._validate_and_extract(full_text, search_start, matches[-1], verbs,

                                              "decimal_with_name", "high")

        # Layer 2.5-BIS: Alpha with name (last occurrence)

        matches = list(re.finditer(alpha_name, search_text, re.IGNORECASE | re.MULTILINE))

        if matches:

            return self._validate_and_extract(full_text, search_start, matches[-1], verbs,

                                              "alpha_with_name", "high")

        # Layer 3: Concluding expressions (MOVED UP)

        matches = list(re.finditer(concluding, search_text, re.IGNORECASE))

        if matches:

            result = self._validate_and_extract(full_text, search_start, matches[0], verbs,

                                                "concluding_expression", "medium-high", require_verbs=True)

            if result:

                return result

        # Layer 4: Acordam expressions

        matches = list(re.finditer(acordam, search_text, re.IGNORECASE))

        if matches:

            return self._validate_and_extract(full_text, search_start, matches[0], verbs,

                                              "acordam", "medium")

        # Layer 5A: Roman isolated (MOVED DOWN)

        matches = list(re.finditer(roman_iso, search_text, re.MULTILINE))

        if matches:

            result = self._validate_and_extract(full_text, search_start, matches[-1], verbs,

                                                "roman_isolated", "medium", require_verbs=True)

            if result:

                return result

        # Layer 5A-BIS: Alpha isolated

        matches = list(re.finditer(alpha_iso, search_text, re.MULTILINE))

        if matches:

            result = self._validate_and_extract(full_text, search_start, matches[-1], verbs,

                                                "alpha_isolated", "medium", require_verbs=True)

            if result:

                return result

        # Layer 5B: Decimal isolated (LOWEST PRIORITY - flag for review)

        matches = list(re.finditer(decimal_iso, search_text, re.MULTILINE))

        if matches:

            result = self._validate_and_extract(full_text, search_start, matches[-1], verbs,

                                                "decimal_isolated", "medium-low", require_verbs=True)

            if result:

                result["metadata"]["requires_manual_review"] = True

                result["metadata"]["review_reason"] = "Isolated decimal numbering - may be fact list - please validate"

            return result

        return None
```


To conclude, we have the auxiliary method _validate_and_extract() that verifies if the extracted decision has the judgmentory verbs.



Final Note:
---
This algorithm obviously could be improved as the regex algorithm may not capture all patterns and fails usually and not commonly structure documents. Then more semantic methods may be more pertinent. But this was the simplest approach.

It was done trough a lot of trial and error, with helpful assistance from AI, altough I as the author take full responsability for the script.