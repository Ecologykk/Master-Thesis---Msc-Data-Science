# Plano de Implementação: App Streamlit de Anotação Judicial (DV/IC)

## Resumo
Construir uma app Streamlit simples, vertical e minimalista, em português de Portugal, para anotação manual por juízes, com:
- Login interno por `username/password` (sem email).
- Fluxo por ramo (DV ou IC), nunca DV+IC no mesmo fluxo do juiz.
- Visualização prioritária do caso por `iFrame` do URL original.
- Classificação + confiança + justificação opcional.
- Gravação manual e autosave a cada 60 segundos.
- Persistência em Google Sheets.
- Botão explícito de fecho final com bloqueio de edição.
- Deploy em duas fases: local primeiro, depois Streamlit Cloud público com login interno.

## Escopo Fechado
- Idioma integral da interface: português de Portugal formal e consistente.
- Layout sem sidebar, coluna única, margens largas, fluxo vertical.
- Design “office antigo”: fundo branco, tipografia serif simples, paleta neutra (azul escuro/cinzento), sem animações e sem efeitos modernos.
- Sem funcionalidades extra fora das que definiste.

## Arquitetura Técnica (separação lógica/interface)
Estrutura proposta em `Master-Thesis---Msc-Data-Science/judge_data_anotation_app`:
- `app.py`: composição de páginas e fluxo Streamlit.
- `ui.py`: componentes visuais (cabeçalho, blocos, botões, mensagens).
- `services/auth.py`: validação de login, hash+sal.
- `services/assignment.py`: mapeamento do ramo por username.
- `services/data_loader.py`: leitura e normalização dos CSV.
- `services/persistence.py`: leitura/escrita Google Sheets.
- `services/progress.py`: cálculo de progresso e regras de conclusão.
- `models.py`: tipos e contratos de dados.
- `styles.css`: estilo minimalista vertical.
- `requirements.txt`: dependências.
- `.streamlit/secrets.toml` (local) e segredos no Cloud.
- `README.md`: instruções de execução/deploy.

## Interfaces Públicas / Contratos de Dados
### 1) Fontes de casos (CSV)
- DV: `data/processed_data/gold_test/dv_gold_test_full.csv`
- IC: `data/processed_data/gold_test/boc_gold_test_full.csv`
- Colunas usadas em runtime:
  - `url`
  - `n_processo`
  - `texto_integral_completo` (apenas fallback)
- Colunas de gold label (`decisao_binaria`, `decisao_ternaria`) nunca são mostradas ao juiz.

### 2) Credenciais (JSON com hash+sal)
Formato lógico:
```json
{
  "J_DV_01": {"salt": "...", "password_hash": "..."},
  "J_IC_01": {"salt": "...", "password_hash": "..."}
}
```
Regra de armazenamento:
- Local: ficheiro JSON `gitignored`.
- Cloud: conteúdo equivalente em `st.secrets` (não no repositório).

### 3) Google Sheets (2 ficheiros separados)
- Ficheiro DV.
- Ficheiro IC.
Cada ficheiro contém:
- Aba `decisao` (matriz `n_processo` x `juiz`)
- Aba `confianca` (matriz `n_processo` x `juiz`)
- Aba `justificacao` (matriz `n_processo` x `juiz`)
- Aba técnica `estado` com colunas: `username`, `ramo`, `finalizado`, `finalizado_em`, `ultima_gravacao_em`

## Regras Funcionais Decididas
### Login e atribuição de ramo
- Prefixo no username:
  - `J_DV_` -> ramo DV
  - `J_IC_` -> ramo IC
- Sem prefixo válido: atribuição “aleatória fixa” por hash determinístico do username (persistente por definição).
- Cada juiz anota apenas 1 ramo (50 casos).

### Classes e confiança
- DV: `Decisão Mantida`, `Decisão Alterada`
- IC: `Decisão Favorável`, `Decisão Desfavorável`, `Decisão Parcial`
- Confiança: `Confiante`, `Não Confiante`
- Justificação: opcional sempre.

### Navegação e progresso
- Ordem inicial dos casos no ramo: aleatória fixa por juiz.
- Navegação livre:
  - Botões `Anterior` / `Seguinte`
  - Slider para qualquer caso do ramo
- Progresso = casos com `decisão` e `confiança` preenchidas / total do ramo.
- Exibir percentagem e contagem explícitas.

### Gravação
- Botão manual `Gravar progresso`.
- Autosave a cada 60s para alterações pendentes.
- Escrita idempotente por célula juiz×caso.
- Feedback visual após gravação (data/hora da última gravação).

### Finalização e bloqueio
- Ao atingir 100%, mostrar botão explícito `Concluir e bloquear anotação`.
- Só após este clique: `estado.finalizado = true` e bloqueio total de edição para esse juiz.
- Antes disso, edição é livre em qualquer caso do ramo.

## Fluxo de UI (vertical, simples)
1. **Página Login**
- Username + password.
- Mensagens claras de erro/sucesso.

2. **Página Instruções**
- Regras objetivas.
- Nota formal para uso de “Não Confiante” em casos-limite/ambíguos.

3. **Página de Anotação**
- Cabeçalho discreto (utilizador, ramo, progresso).
- Janela principal com `st.components.v1.iframe(url)`.
- Se a visualização não for utilizável: fallback manual para texto local do CSV.
- Bloco de classificação (botões grandes).
- Bloco de confiança (botões grandes).
- Bloco de justificação opcional.
- Bloco de ações: gravar, anterior, seguinte.

4. **Página Final**
- Agradecimento formal.
- Estado fechado (apenas leitura).

## Deploy
### Fase 1 (local)
- Executar localmente com credenciais JSON `gitignored`.
- Validar fluxo completo com juízes piloto.

### Fase 2 (Streamlit Cloud)
- App pública com login interno da app.
- Segredos configurados no painel do Streamlit (`Google service account`, IDs dos ficheiros, credenciais).
- Sem necessidade de conta/email Streamlit para juízes.

## Plano de Testes
### Testes funcionais
- Login válido/inválido.
- Roteamento DV/IC por prefixo.
- Atribuição fixa para username sem prefixo.
- Renderização de classes corretas por ramo.
- Progresso correto com edição livre.
- Gravação manual e autosave.
- Bloqueio após finalização explícita.
- Reentrada após logout com estado preservado.

### Testes de dados/persistência
- Escrita correta nas 3 matrizes.
- Criação de coluna nova para juiz novo.
- Integridade por `n_processo` (sem trocas de linha).
- Estado finalizado persistente em aba `estado`.

### Testes de fallback de visualização
- Caso normal com iFrame.
- Caso com iFrame não utilizável -> fallback texto local.

## Critérios de Aceitação
- Juiz consegue anotar o seu ramo completo sem apoio técnico.
- UX mantém-se simples e linear (sem sidebar, sem elementos escondidos).
- Nenhuma label gold é exposta.
- Não há perda de trabalho com autosave de 60s.
- Finalização bloqueia edição de forma persistente.
- Deploy Cloud acessível sem contas externas dos juízes.

## Assunções e Defaults Explícitos
- Formato de ingestão: CSV (`utf-8`) por simplicidade e robustez.
- Fallback da v1: `iFrame + texto local` (sem HTML/Selenium nesta fase).
- Ordem inicial dentro do ramo: aleatória fixa por juiz.
- Abordagem de segurança: hash+sal e segredos fora do repositório no Cloud.

## Referências externas usadas para decisão de acesso no Streamlit Cloud
- https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app
- https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app/invite-your-friends
