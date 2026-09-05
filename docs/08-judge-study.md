# 08 — Judge Annotation Study

## Purpose
A Streamlit web app ([`judge_labeling_app/`](../judge_labeling_app/)) used to collect manual
outcome annotations from Portuguese judges on the 50-case gold test set per case type (DV, BoC).
It exists to validate the automated label-extraction logic against expert human judgment and to
give a human baseline to compare model predictions against. It is a standalone app, separate from
the modelling pipelines in [04-bert.md](04-bert.md)/[05-llm.md](05-llm.md), with its own
persistence layer (Google Sheets, not local files).

## Prerequisites
- Python deps from [`judge_labeling_app/requirements.txt`](../judge_labeling_app/requirements.txt)
  (includes `streamlit`, `streamlit_autorefresh`, a Google Sheets client).
- Two Google Sheets spreadsheets already created (one for DV, one for IC/BoC) and a Google service
  account with edit access to both, configured via `.streamlit/secrets.toml`.
- Judge login credentials configured in `credentials/juizes.json` (or another path pointed to by
  `JUDGE_CREDENTIALS_PATH`).
- `data/processed_data/gold_test/{dv,boc}_gold_test_full.csv` present (produced by
  [03-data-processing.md](03-data-processing.md)) — the app reads `url`, `n_processo`, and
  `texto_integral_completo` from these files at runtime. Note `texto_integral_sem_decisao` (the
  decision-redacted column used everywhere else in the pipeline) is **not** used here — judges see
  the full original text, including the actual outcome, since they are producing the ground truth
  the model outputs get compared against, not blind-predicting from redacted text.

## Exact commands
All from the `judge_labeling_app/` directory:
```bash
cd judge_labeling_app
pip install -r requirements.txt

cp .streamlit/secrets.example.toml .streamlit/secrets.toml
# then edit secrets.toml: gsheets_dv / gsheets_ic connections + service-account credentials

cp credentials/judge_credentials.example.json credentials/juizes.json
# then edit juizes.json: real usernames/passwords

streamlit run app.py
```
This launches a local Streamlit server (default `http://localhost:8501`); see
[`judge_labeling_app/README.md`](../judge_labeling_app/README.md) for the Streamlit Cloud deploy
variant (public app + internal username/password login, secrets configured in the Cloud
dashboard's Secrets panel instead of a local file).

## Inputs
- `data/processed_data/gold_test/{dv,boc}_gold_test_full.csv` (columns: `url`, `n_processo`,
  `texto_integral_completo`).
- `.streamlit/secrets.toml` — Google Sheets connection details + service-account credentials.
- `credentials/juizes.json` (or `JUDGE_CREDENTIALS_PATH`) — judge login credentials. Supported
  formats: `{"J_DV_01": {"password": "1234"}}`, `{"J_DV_01": "1234"}`, or the legacy
  `{"J_DV_01": {"salt": "...", "password_hash": "..."}}`.
- Each judge's existing responses/state, read back from the Google Sheets on login (so sessions
  resume where they left off).

## Outputs
Judge responses are written **only to Google Sheets**, never to local files — two spreadsheets
(DV and IC), each with tabs:
- `decisao`, `confianca`, `justificacao` — matrices of `n_processo` × `juiz`.
- `estado` — per-judge state: `username`, `ramo` (assigned branch), `finalizado`,
  `finalizado_em`, `ultima_gravacao_em`.

Per the app's own README, the collected judge responses and any analysis of them are explicitly
excluded from the public repository:
- `reports/*.csv` (judge decisions/confidence/justifications) — excluded.
- `reports/*.ipynb` (analysis notebooks over those responses) — excluded.
- `.streamlit/secrets.toml` and `credentials/juizes.json` (real credentials) — excluded; only the
  `.example` templates are versioned.
- The application source code itself, and the credential/secrets *templates*, are included.

## Approximate runtime
Not a batch job — an interactive multi-session web app. Each judge works through up to 50 cases
at their own pace, with autosave every 60 seconds and a manual "Gravar progresso" button; there is
no fixed completion time to estimate.

## Expected result
On login, the branch (DV vs IC) is auto-assigned from the username prefix (`J_DV_*` → DV,
`J_IC_*` → IC, otherwise a fixed pseudo-random hash of the username) — see
`services/assignment.py`. The judge reads each case (embedded via `iframe` to the original court
URL, with a manual fallback to `texto_integral_completo` if the URL isn't viewable or is missing),
selects an outcome class, indicates confidence, optionally writes a justification, and can
navigate freely between cases (Previous/Next or a direct-selection slider). A progress indicator
shows `x/50 (y%)` with a visual dot strip per case. Once all 50 cases have a decision + confidence
recorded, a "Concluir e bloquear anotação" button appears; clicking it writes `finalizado=True`
to the `estado` tab and locks the session (further visits show a read-only "submission complete"
page).

This doc does not cover hardware-determinism (docs/09) — human annotation has no such caveat.
