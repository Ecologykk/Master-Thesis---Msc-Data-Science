# App de Anotação Judicial (DV/IC)

Aplicação Streamlit para anotação manual por juízes, com foco em simplicidade de uso, fluxo vertical e persistência em Google Sheets.

## Funcionalidades implementadas
- Interface integral em português de Portugal, com linguagem formal.
- Login interno por `username/password` (sem email).
- Atribuição automática de ramo:
  - `J_DV_*` -> Violência Doméstica (DV)
  - `J_IC_*` -> Incumprimento Contratual (IC)
  - sem prefixo -> atribuição pseudoaleatória fixa por hash do `username`
- Workflow vertical:
  1. Leitura do caso (website por `iFrame`)
  2. Escolha da classe de desfecho
  3. Indicação de confiança
  4. Justificação opcional
  5. Gravação manual / autosave
- Fallback de visualização:
  - prioridade: `iFrame` com URL original
  - fallback: `texto_integral_completo`
- Navegação livre entre casos:
  - `Anterior` / `Seguinte`
  - seleção direta por slider
- Progresso explícito (`x/50` e percentagem), com indicador visual de casos.
- Gravação:
  - botão manual `Gravar progresso`
  - autosave a cada 60 segundos (se houver alterações pendentes)
- Bloqueio final:
  - botão explícito `Concluir e bloquear anotação` após 100%
  - estado persistente em aba técnica `estado`

## Estrutura
```text
judge_data_anotation_app/
  app.py
  models.py
  ui.py
  styles.css
  requirements.txt
  services/
    auth.py
    assignment.py
    data_loader.py
    persistence.py
    progress.py
  credentials/
    judge_credentials.example.json
  .streamlit/
    secrets.example.toml
```

## Dados de estudo: o que foi recolhido e o que foi excluído
Esta app foi usada para recolher anotações manuais de juízes sobre um subconjunto dos casos
(o "gold test set"), para validar a lógica de extração de rótulos e comparar com as previsões
dos modelos. Isto envolve dados de participantes de estudo identificáveis (juízes), o que é
distinto do texto dos casos em si (pseudonimizado na origem pelas bases de dados públicas).

**Excluído do repositório público** (dados de participantes do estudo):
- As respostas dos juízes (decisão, confiança, justificação) — ficheiros em `reports/*.csv`.
- Quaisquer notebooks de análise dessas respostas — `reports/*.ipynb`.
- Credenciais reais de login e da service account Google — `.streamlit/secrets.toml`,
  `credentials/*.json` (exceto os templates `.example`).

**Incluído** (código-fonte e templates apenas):
- O código-fonte completo da aplicação.
- Templates com placeholders para as credenciais (`credentials/judge_credentials.example.json`,
  `.streamlit/secrets.example.toml`) — copiar e preencher com valores reais para reproduzir.

## Dados utilizados
Localização dos datasets:
- `../data/processed_data/gold_test/dv_gold_test_full.csv`
- `../data/processed_data/gold_test/boc_gold_test_full.csv`

Colunas usadas em runtime:
- `url`
- `n_processo`
- `texto_integral_completo`

`texto_integral_sem_decisao` não é usado pela aplicação.

## Modelo de persistência (Google Sheets)
Existem **2 ficheiros** de Google Sheets: um para DV e outro para IC.

Cada ficheiro contém:
- aba `decisao` (matriz `n_processo` x `juiz`)
- aba `confianca` (matriz `n_processo` x `juiz`)
- aba `justificacao` (matriz `n_processo` x `juiz`)
- aba `estado` com:
  - `username`
  - `ramo`
  - `finalizado`
  - `finalizado_em`
  - `ultima_gravacao_em`

Nota:
- A app tenta atualizar a aba e, se não existir, tenta criá-la automaticamente.

## Configuração local
1. Instalar dependências:
```bash
pip install -r requirements.txt
```

2. Copiar `.streamlit/secrets.example.toml` para `.streamlit/secrets.toml` e preencher com
   valores reais:
- ligações Google Sheets (`gsheets_dv` e `gsheets_ic`, ou nomes custom com `gsheets_connection_dv` / `gsheets_connection_ic`)
- credenciais da service account Google

3. Configurar credenciais de login dos juízes:
- copiar `credentials/judge_credentials.example.json` para `credentials/juizes.json`
  (ou outro caminho, apontado via `credentials_path` em `secrets.toml` ou a variável de
  ambiente `JUDGE_CREDENTIALS_PATH`)
- ajustar utilizadores e palavras-passe
- manter o ficheiro real fora de controlo de versão (já coberto por `.gitignore`)

4. Executar:
```bash
streamlit run app.py
```

## Deploy no Streamlit Cloud
Modelo recomendado para este caso:
- app pública
- controlo de acesso feito por login interno na app

Passos:
1. Publicar este diretório no repositório.
2. No painel da app no Streamlit Cloud, configurar `Secrets` com conteúdo equivalente ao `secrets.toml`.
3. Definir credenciais de login:
  - recomendado: `credentials_json` em `Secrets`
  - alternativa: `auth_credentials` em `Secrets`
4. Validar:
  - login
  - leitura por iFrame e fallback de texto
  - gravação manual/autosave
  - bloqueio final

## Formatos de credenciais suportados
- Recomendado (simples): `{ "J_DV_01": { "password": "1234" } }`
- Também suportado: `{ "J_DV_01": "1234" }`
- Compatibilidade antiga: `{ "J_DV_01": { "salt": "...", "password_hash": "..." } }`

## Notas de segurança
- Em ambiente de produção, evitar passwords em texto simples.
- Preferir hash+sal por utilizador.
- Não commitar:
  - `.streamlit/secrets.toml`
  - `credentials/juizes.json`

