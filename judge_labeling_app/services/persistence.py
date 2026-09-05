from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from models import Ramo

try:
    from streamlit_gsheets import GSheetsConnection
except ImportError as exc:  # pragma: no cover - depende do ambiente
    GSheetsConnection = None
    GSHEETS_IMPORT_ERROR = exc
else:
    GSHEETS_IMPORT_ERROR = None


MATRIX_WORKSHEETS = ("decisao", "confianca", "justificacao")
STATE_WORKSHEET = "estado"
STATE_COLUMNS = [
    "username",
    "ramo",
    "finalizado",
    "finalizado_em",
    "ultima_gravacao_em",
]


class PersistenceError(RuntimeError):
    pass


def now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class GSheetsPersistence:
    def __init__(self, branch: Ramo):
        if GSheetsConnection is None:
            raise PersistenceError(
                "Dependência 'st-gsheets-connection' não encontrada. "
                f"Erro original: {GSHEETS_IMPORT_ERROR}"
            )

        self.branch = branch
        self.connection_name = self._resolve_connection_name(branch)
        try:
            self.connection = st.connection(
                self.connection_name, type=GSheetsConnection
            )
        except Exception as exc:  # pragma: no cover - depende da configuração local
            raise PersistenceError(
                f"Não foi possível abrir a ligação Google Sheets '{self.connection_name}'."
            ) from exc

    @staticmethod
    def _resolve_connection_name(branch: Ramo) -> str:
        default = "gsheets_dv" if branch == Ramo.DV else "gsheets_ic"
        config_key = (
            "gsheets_connection_dv" if branch == Ramo.DV else "gsheets_connection_ic"
        )
        configured = str(st.secrets.get(config_key, "")).strip()
        return configured or default

    def _read_worksheet(self, worksheet: str) -> pd.DataFrame:
        try:
            data = self.connection.read(worksheet=worksheet, ttl=0)
        except Exception:
            return pd.DataFrame()
        if data is None:
            return pd.DataFrame()
        if not isinstance(data, pd.DataFrame):
            return pd.DataFrame(data)
        cleaned = data.copy()
        cleaned = cleaned.loc[
            :,
            ~cleaned.columns.astype(str).str.startswith("Unnamed"),
        ]
        return cleaned

    def _write_worksheet(self, worksheet: str, df: pd.DataFrame) -> None:
        def _format_error(exc: Exception) -> str:
            message = str(exc).strip()
            name = type(exc).__name__
            return f"{name}: {message}" if message else name

        try:
            self.connection.update(worksheet=worksheet, data=df)
            return
        except Exception as update_exc:  # pragma: no cover - depende da rede/segredos
            # Se a folha não existir ainda, tenta criá-la e gravar de uma só vez.
            try:
                self.connection.create(worksheet=worksheet, data=df)
                return
            except Exception as create_exc:
                raise PersistenceError(
                    (
                        f"Falha ao gravar na folha '{worksheet}' ({self.connection_name}). "
                        f"Erro update: {_format_error(update_exc)}. "
                        f"Erro create: {_format_error(create_exc)}."
                    )
                ) from create_exc

    @staticmethod
    def _normalize_case_ids(case_ids: list[str]) -> list[str]:
        normalized = [
            str(case_id).strip() for case_id in case_ids if str(case_id).strip()
        ]
        return list(dict.fromkeys(normalized))

    def _normalize_matrix(
        self,
        df: pd.DataFrame,
        case_ids: list[str],
        username: str,
    ) -> pd.DataFrame:
        normalized_case_ids = self._normalize_case_ids(case_ids)
        base = pd.DataFrame({"n_processo": normalized_case_ids})

        if df.empty or "n_processo" not in df.columns:
            result = base
        else:
            matrix = df.copy()
            matrix["n_processo"] = matrix["n_processo"].astype(str).str.strip()
            matrix = matrix[matrix["n_processo"] != ""]
            matrix = matrix.drop_duplicates(subset=["n_processo"], keep="first")
            extra_columns = [col for col in matrix.columns if col != "n_processo"]
            result = base.merge(
                matrix[["n_processo", *extra_columns]], on="n_processo", how="left"
            )

        if username not in result.columns:
            result[username] = ""

        result = result.fillna("")
        return result

    def load_user_responses(
        self,
        username: str,
        case_ids: list[str],
    ) -> dict[str, dict[str, str]]:
        responses: dict[str, dict[str, str]] = {
            case_id: {"decisao": "", "confianca": "", "justificacao": ""}
            for case_id in self._normalize_case_ids(case_ids)
        }

        mapping = {
            "decisao": "decisao",
            "confianca": "confianca",
            "justificacao": "justificacao",
        }
        for worksheet, field in mapping.items():
            matrix = self._normalize_matrix(
                self._read_worksheet(worksheet), case_ids, username
            )
            for row in matrix.itertuples(index=False):
                case_id = str(getattr(row, "n_processo", "")).strip()
                if not case_id:
                    continue
                value = str(getattr(row, username, "")).strip()
                if case_id in responses:
                    responses[case_id][field] = value
        return responses

    def save_user_responses(
        self,
        username: str,
        case_ids: list[str],
        responses: dict[str, dict[str, str]],
    ) -> None:
        field_by_worksheet = {
            "decisao": "decisao",
            "confianca": "confianca",
            "justificacao": "justificacao",
        }
        normalized_case_ids = self._normalize_case_ids(case_ids)

        for worksheet, field in field_by_worksheet.items():
            matrix = self._normalize_matrix(
                self._read_worksheet(worksheet), normalized_case_ids, username
            )
            values = []
            for case_id in matrix["n_processo"].tolist():
                value = responses.get(case_id, {}).get(field, "")
                values.append(str(value).strip())
            matrix[username] = values
            self._write_worksheet(worksheet, matrix)

    def _normalize_state_table(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=STATE_COLUMNS)

        state_df = df.copy()
        for column in STATE_COLUMNS:
            if column not in state_df.columns:
                state_df[column] = ""
        state_df = state_df[STATE_COLUMNS]
        state_df = state_df.fillna("")
        return state_df

    def load_user_state(self, username: str) -> dict[str, Any]:
        state_df = self._normalize_state_table(self._read_worksheet(STATE_WORKSHEET))
        if state_df.empty:
            return {
                "finalizado": False,
                "finalizado_em": "",
                "ultima_gravacao_em": "",
            }

        cleaned_username = username.strip()
        mask = (state_df["username"] == cleaned_username) & (
            state_df["ramo"] == self.branch.value
        )
        if not mask.any():
            return {
                "finalizado": False,
                "finalizado_em": "",
                "ultima_gravacao_em": "",
            }

        row = state_df[mask].iloc[0]
        final_raw = str(row["finalizado"]).strip().lower()
        is_final = final_raw in {"1", "true", "sim", "yes"}
        return {
            "finalizado": is_final,
            "finalizado_em": str(row["finalizado_em"]).strip(),
            "ultima_gravacao_em": str(row["ultima_gravacao_em"]).strip(),
        }

    def upsert_user_state(
        self,
        username: str,
        *,
        finalizado: bool | None = None,
        finalizado_em: str | None = None,
        ultima_gravacao_em: str | None = None,
    ) -> None:
        state_df = self._normalize_state_table(self._read_worksheet(STATE_WORKSHEET))
        cleaned_username = username.strip()
        mask = (state_df["username"] == cleaned_username) & (
            state_df["ramo"] == self.branch.value
        )

        if not mask.any():
            new_row = {
                "username": cleaned_username,
                "ramo": self.branch.value,
                "finalizado": "",
                "finalizado_em": "",
                "ultima_gravacao_em": "",
            }
            state_df = pd.concat([state_df, pd.DataFrame([new_row])], ignore_index=True)
            mask = (state_df["username"] == cleaned_username) & (
                state_df["ramo"] == self.branch.value
            )

        row_index = state_df.index[mask][0]

        if finalizado is not None:
            state_df.at[row_index, "finalizado"] = "true" if finalizado else "false"
        if finalizado_em is not None:
            state_df.at[row_index, "finalizado_em"] = finalizado_em
        if ultima_gravacao_em is not None:
            state_df.at[row_index, "ultima_gravacao_em"] = ultima_gravacao_em

        self._write_worksheet(STATE_WORKSHEET, state_df)
