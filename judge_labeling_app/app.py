from __future__ import annotations

import time
import base64
from pathlib import Path

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from models import OPCOES_CONFIANCA, OPCOES_DECISAO, ROTULOS_RAMO, Ramo
from services.assignment import assign_branch, build_fixed_case_order
from services.auth import load_credentials, verify_login
from services.data_loader import load_cases
from services.persistence import GSheetsPersistence, PersistenceError, now_timestamp
from services.progress import compute_progress, first_incomplete_index
from ui import (
    load_css,
    render_case_block,
    render_choice_buttons,
    render_page_title,
    render_save_feedback,
    render_top_header,
)

APP_DIR = Path(__file__).resolve().parent
AUTO_SAVE_SECONDS = 60


def _resolve_logo_path(*file_names: str) -> Path | None:
    for file_name in file_names:
        candidate = APP_DIR / file_name
        if candidate.exists():
            return candidate
    return None


LOGO_FCUL_PATH = _resolve_logo_path(
    "Logo_Faculdade_Ciências_Lisboa.png",
)
LOGO_CSM_PATH = _resolve_logo_path(
    "Logo-CSM-curto-e1554767364818.jpg",
    "Logo-CSM-curto-e1554767364818.png",
)

PAGE_LOGIN = "login"
PAGE_INSTRUCOES = "instrucoes"
PAGE_ANOTACAO = "anotacao"
PAGE_FINAL = "final"


st.set_page_config(
    page_title="Anotação Judicial",
    layout="centered",
)
load_css(APP_DIR / "styles.css")


def _init_session() -> None:
    defaults = {
        "authenticated": False,
        "username": "",
        "branch": "",
        "page": PAGE_LOGIN,
        "responses": {},
        "ordered_case_ids": [],
        "current_index": 0,
        "dirty": False,
        "last_mutation_epoch": 0.0,
        "last_saved_epoch": 0.0,
        "last_save_message": "",
        "last_save_kind": "info",
        "finalized": False,
        "finalized_at": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_auth_state() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


@st.cache_data(show_spinner=False)
def _load_cases_cached(branch_value: str):
    return load_cases(Ramo(branch_value))


def _cases_by_branch(branch: Ramo):
    cases = _load_cases_cached(branch.value)
    return cases, {case.n_processo: case for case in cases}


def _get_store(branch: Ramo) -> GSheetsPersistence:
    return GSheetsPersistence(branch)


def _touch_dirty() -> None:
    st.session_state["dirty"] = True
    st.session_state["last_mutation_epoch"] = time.time()


def _scroll_to_top() -> None:
    st.components.v1.html(
        """
        <script>
        const jumpTop = () => {
          try {
            window.parent.scrollTo({ top: 0, left: 0, behavior: "auto" });
            const doc = window.parent.document;
            doc.documentElement.scrollTop = 0;
            doc.body.scrollTop = 0;
            const app = doc.querySelector('[data-testid="stAppViewContainer"]');
            if (app) app.scrollTop = 0;
          } catch (e) {}
        };
        jumpTop();
        setTimeout(jumpTop, 30);
        setTimeout(jumpTop, 120);
        setTimeout(jumpTop, 300);
        </script>
        """,
        height=0,
    )


def _render_footer_logos() -> None:
    st.markdown("---")
    logo_tags: list[str] = []

    if LOGO_FCUL_PATH:
        fcul_base64 = base64.b64encode(LOGO_FCUL_PATH.read_bytes()).decode("ascii")
        logo_tags.append(
            "<img "
            f"src='data:image/png;base64,{fcul_base64}' "
            "alt='Logo Faculdade de Ciências da Universidade de Lisboa' "
            "style='max-width:260px; width:100%; height:auto; object-fit:contain;' />"
        )

    if LOGO_CSM_PATH:
        csm_mime = (
            "image/png" if LOGO_CSM_PATH.suffix.lower() == ".png" else "image/jpeg"
        )
        csm_base64 = base64.b64encode(LOGO_CSM_PATH.read_bytes()).decode("ascii")
        logo_tags.append(
            "<img "
            f"src='data:{csm_mime};base64,{csm_base64}' "
            "alt='Logo Conselho Superior da Magistratura' "
            "style='max-width:260px; width:100%; height:auto; object-fit:contain;' />"
        )

    if logo_tags:
        st.markdown(
            (
                "<div style='display:flex; justify-content:space-between; "
                "align-items:flex-end; gap:1rem; width:100%;'>"
                + "".join(logo_tags)
                + "</div>"
            ),
            unsafe_allow_html=True,
        )


def _pending_case_positions(
    ordered_case_ids: list[str],
    responses: dict[str, dict[str, str]],
) -> list[int]:
    pending: list[int] = []
    for idx, case_id in enumerate(ordered_case_ids, start=1):
        response = responses.get(case_id, {})
        has_decision = bool(str(response.get("decisao", "")).strip())
        has_confidence = bool(str(response.get("confianca", "")).strip())
        if not (has_decision and has_confidence):
            pending.append(idx)
    return pending


def _render_case_status_indicator(
    total_cases: int,
    current_position: int,
    pending_positions: list[int],
) -> None:
    pending_set = set(pending_positions)
    dots: list[str] = []
    for pos in range(1, total_cases + 1):
        classes = ["case-dot"]
        if pos == current_position:
            classes.append("current")
        elif pos in pending_set:
            classes.append("pending")
        else:
            classes.append("done")
        dots.append(f"<span class='{' '.join(classes)}' title='Caso {pos}'></span>")

    st.markdown(
        "<div class='case-status-strip'>" + "".join(dots) + "</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "Legenda dos quadrados: azul escuro = caso atual; amarelo = por completar; verde = completo."
    )


def _update_response(case_id: str, field: str, value: str) -> None:
    case_responses = st.session_state["responses"].setdefault(
        case_id,
        {"decisao": "", "confianca": "", "justificacao": ""},
    )
    new_value = value.strip()
    if case_responses.get(field, "") == new_value:
        return
    case_responses[field] = new_value
    _touch_dirty()


def _save_progress(force: bool = False) -> bool:
    if not st.session_state.get("authenticated", False):
        return False

    is_dirty = st.session_state.get("dirty", False)
    if not force and not is_dirty:
        return True

    username = st.session_state["username"]
    branch = Ramo(st.session_state["branch"])
    ordered_case_ids = st.session_state["ordered_case_ids"]
    responses = st.session_state["responses"]
    timestamp = now_timestamp()

    try:
        store = _get_store(branch)
        store.save_user_responses(username, ordered_case_ids, responses)
        store.upsert_user_state(username, ultima_gravacao_em=timestamp)
    except PersistenceError as exc:
        st.session_state["last_save_message"] = f"Falha ao gravar: {exc}"
        st.session_state["last_save_kind"] = "error"
        return False

    st.session_state["dirty"] = False
    st.session_state["last_saved_epoch"] = time.time()
    st.session_state["last_save_message"] = f"Progresso gravado em {timestamp}."
    st.session_state["last_save_kind"] = "success"
    return True


def _finish_annotation() -> None:
    if not _save_progress(force=True):
        return

    username = st.session_state["username"]
    branch = Ramo(st.session_state["branch"])
    final_timestamp = now_timestamp()

    try:
        store = _get_store(branch)
        store.upsert_user_state(
            username,
            finalizado=True,
            finalizado_em=final_timestamp,
            ultima_gravacao_em=final_timestamp,
        )
    except PersistenceError as exc:
        st.session_state["last_save_message"] = f"Falha ao finalizar: {exc}"
        st.session_state["last_save_kind"] = "error"
        return

    st.session_state["finalized"] = True
    st.session_state["finalized_at"] = final_timestamp
    st.session_state["page"] = PAGE_FINAL
    st.rerun()


def _initialize_user_runtime(username: str, branch: Ramo) -> None:
    cases, _ = _cases_by_branch(branch)
    case_ids = [case.n_processo for case in cases]
    ordered_case_ids = build_fixed_case_order(case_ids, username, branch)

    store = _get_store(branch)
    loaded_responses = store.load_user_responses(username, case_ids)
    state = store.load_user_state(username)

    responses = {
        case_id: {
            "decisao": loaded_responses.get(case_id, {}).get("decisao", ""),
            "confianca": loaded_responses.get(case_id, {}).get("confianca", ""),
            "justificacao": loaded_responses.get(case_id, {}).get("justificacao", ""),
        }
        for case_id in case_ids
    }

    st.session_state["authenticated"] = True
    st.session_state["username"] = username
    st.session_state["branch"] = branch.value
    st.session_state["responses"] = responses
    st.session_state["ordered_case_ids"] = ordered_case_ids
    st.session_state["current_index"] = first_incomplete_index(
        ordered_case_ids, responses
    )
    st.session_state["dirty"] = False
    st.session_state["last_mutation_epoch"] = 0.0
    st.session_state["last_saved_epoch"] = time.time()
    st.session_state["last_save_message"] = ""
    st.session_state["last_save_kind"] = "info"
    st.session_state["finalized"] = bool(state.get("finalizado", False))
    st.session_state["finalized_at"] = str(state.get("finalizado_em", "")).strip()
    st.session_state["page"] = (
        PAGE_FINAL if st.session_state["finalized"] else PAGE_INSTRUCOES
    )


def _render_login_page() -> None:
    render_page_title(
        "Plataforma de Anotação Judicial",
        "Introduza as suas credenciais para iniciar sessão.",
    )

    credentials = load_credentials(APP_DIR)
    if not credentials:
        st.error(
            "Não foram encontradas credenciais válidas. "
            "Confirme a configuração do ficheiro de credenciais."
        )
        _render_footer_logos()
        return

    with st.form("form_login", clear_on_submit=False):
        username = st.text_input("Nome de utilizador")
        password = st.text_input("Palavra-passe", type="password")
        submit = st.form_submit_button(
            "Entrar",
            use_container_width=True,
            key="btn_login_submit",
        )

    if not submit:
        _render_footer_logos()
        return

    cleaned_username = username.strip()
    if not verify_login(cleaned_username, password, credentials):
        st.error("Credenciais inválidas. Verifique os dados e tente novamente.")
        _render_footer_logos()
        return

    branch = assign_branch(cleaned_username)
    try:
        _initialize_user_runtime(cleaned_username, branch)
    except (FileNotFoundError, ValueError, PersistenceError) as exc:
        st.error(f"Não foi possível iniciar a sessão: {exc}")
        _render_footer_logos()
        return

    st.rerun()


def _render_instruction_page() -> None:
    branch = Ramo(st.session_state["branch"])
    branch_label = ROTULOS_RAMO[branch]
    render_page_title("Instruções", "Leia atentamente antes de iniciar.")

    st.markdown(
        (
            "### Objetivo da Plataforma\n"
            "Esta plataforma destina-se à anotação manual de decisões judiciais.\n\n"
            f"**Ramo atribuído nesta sessão:** {branch_label}.\n\n"
            "### Fluxo de Trabalho por Caso\n"
            "1. Leia o caso com atenção.\n"
            "2. Selecione a classe de desfecho adequada.\n"
            "3. Indique o grau de confiança da sua classificação.\n"
            "4. Escreva uma justificação, se necessário.\n"
            "5. Grave o progresso e avance para o caso seguinte.\n\n"
            '### Quando selecionar "Não Confiante"\n'
            "- Quando o caso estiver no limite entre duas classes possíveis.\n"
            "- Quando a fundamentação for ambígua ou insuficiente para uma classificação segura.\n"
            "- Quando existirem dúvidas significativas sobre a adequação da classe escolhida."
        )
    )

    start_col, logout_col = st.columns(2)
    with start_col:
        if st.button("Iniciar anotação", use_container_width=True, type="primary"):
            st.session_state["page"] = PAGE_ANOTACAO
            st.rerun()
    with logout_col:
        if st.button(
            "Terminar sessão",
            use_container_width=True,
            key="btn_logout_instructions",
        ):
            _clear_auth_state()
            _init_session()
            st.rerun()

    _render_footer_logos()


def _render_annotation_page() -> None:
    st_autorefresh(interval=AUTO_SAVE_SECONDS * 1000, key="autosave_tick")
    if st.session_state.pop("scroll_to_top_pending", False):
        _scroll_to_top()

    username = st.session_state["username"]
    branch = Ramo(st.session_state["branch"])
    ordered_case_ids = st.session_state["ordered_case_ids"]
    responses = st.session_state["responses"]

    cases, case_map = _cases_by_branch(branch)
    if not cases:
        st.error("Não existem casos disponíveis para anotação.")
        return

    ordered_case_ids = [case_id for case_id in ordered_case_ids if case_id in case_map]
    if not ordered_case_ids:
        st.error("Não foi possível carregar os identificadores dos casos.")
        return
    st.session_state["ordered_case_ids"] = ordered_case_ids

    answered, total, progress = compute_progress(ordered_case_ids, responses)
    progress_text = f"{answered}/{total} ({progress * 100:.0f}%)"
    render_top_header(username, ROTULOS_RAMO[branch], progress_text)
    render_page_title(
        "Anotação de Casos",
        "Analise cada caso e escolha a opção pretendida.",
    )
    st.progress(progress, text=f"{answered} de {total} casos concluídos.")
    st.info(
        "Nota operacional: para garantir o registo visual da escolha, clique duas vezes no botão pretendido."
    )

    if st.button("Ver resumo rápido das instruções", use_container_width=True):
        st.session_state["show_instructions_summary"] = not st.session_state.get(
            "show_instructions_summary", False
        )
    if st.session_state.get("show_instructions_summary", False):
        st.info(
            "Resumo: leia o caso, selecione a classe, indique confiança, justifique se necessário e grave o progresso. Use 'Não Confiante' em casos-limite, ambíguos ou com dúvida relevante."
        )

    current_index = min(st.session_state["current_index"], len(ordered_case_ids) - 1)
    st.session_state["current_index"] = current_index

    slider_value = st.select_slider(
        "Selecionar caso",
        options=list(range(1, len(ordered_case_ids) + 1)),
        value=current_index + 1,
    )
    st.session_state["current_index"] = slider_value - 1
    current_index = st.session_state["current_index"]
    pending_positions = _pending_case_positions(ordered_case_ids, responses)
    _render_case_status_indicator(
        total_cases=len(ordered_case_ids),
        current_position=current_index + 1,
        pending_positions=pending_positions,
    )

    current_case_id = ordered_case_ids[current_index]
    current_case = case_map[current_case_id]
    current_response = responses.setdefault(
        current_case_id,
        {"decisao": "", "confianca": "", "justificacao": ""},
    )

    fallback_key = f"fallback_{current_case_id}"
    if fallback_key not in st.session_state:
        st.session_state[fallback_key] = False
    if not current_case.url:
        st.session_state[fallback_key] = True
        st.warning("URL não disponível para este caso. Foi ativado o texto integral.")

    st.checkbox(
        "Se o website não estiver visível, ative o texto integral do caso.",
        key=fallback_key,
    )
    render_case_block(current_case, bool(st.session_state[fallback_key]))

    new_decision = render_choice_buttons(
        "Classificação do desfecho",
        OPCOES_DECISAO[branch],
        current_response.get("decisao", ""),
        key_prefix=f"decisao_{current_case_id}",
    )
    if new_decision != current_response.get("decisao", ""):
        _update_response(current_case_id, "decisao", new_decision)

    new_confidence = render_choice_buttons(
        "Grau de confiança",
        OPCOES_CONFIANCA,
        current_response.get("confianca", ""),
        key_prefix=f"confianca_{current_case_id}",
    )
    if new_confidence != current_response.get("confianca", ""):
        _update_response(current_case_id, "confianca", new_confidence)

    st.markdown("### Justificação (opcional)")
    justification_key = f"justificacao_{current_case_id}"
    if justification_key not in st.session_state:
        st.session_state[justification_key] = current_response.get("justificacao", "")
    justification_text = st.text_area(
        "Introduza a justificação, se necessário.",
        key=justification_key,
        height=160,
        label_visibility="collapsed",
    )
    if justification_text != current_response.get("justificacao", ""):
        _update_response(current_case_id, "justificacao", justification_text)

    render_save_feedback(
        st.session_state.get("last_save_message", ""),
        st.session_state.get("last_save_kind", "info"),
    )

    nav_col_1, nav_col_2, nav_col_3 = st.columns(3)
    with nav_col_1:
        if st.button(
            "Anterior",
            use_container_width=True,
            disabled=current_index == 0,
        ):
            st.session_state["current_index"] = current_index - 1
            st.session_state["scroll_to_top_pending"] = True
            st.rerun()
    with nav_col_2:
        if st.button("Gravar progresso", use_container_width=True, type="primary"):
            _save_progress(force=True)
            st.rerun()
    with nav_col_3:
        if st.button(
            "Seguinte",
            use_container_width=True,
            disabled=current_index >= (len(ordered_case_ids) - 1),
        ):
            st.session_state["current_index"] = current_index + 1
            st.session_state["scroll_to_top_pending"] = True
            st.rerun()

    answered, total, progress = compute_progress(ordered_case_ids, responses)
    if answered == total:
        st.warning(
            "Todos os casos foram preenchidos. Se estiver concluído, finalize para bloquear a anotação."
        )
        if st.button("Concluir e bloquear anotação", use_container_width=True):
            _finish_annotation()

    if st.button(
        "Terminar sessão",
        use_container_width=True,
        key="btn_logout_annotation",
    ):
        _clear_auth_state()
        _init_session()
        st.rerun()

    now = time.time()
    dirty = st.session_state.get("dirty", False)
    last_mutation = st.session_state.get("last_mutation_epoch", 0.0)
    if dirty and (now - last_mutation) >= AUTO_SAVE_SECONDS:
        _save_progress(force=True)


def _render_final_page() -> None:
    branch = Ramo(st.session_state["branch"])
    ordered_case_ids = st.session_state["ordered_case_ids"]
    responses = st.session_state["responses"]
    answered, total, progress = compute_progress(ordered_case_ids, responses)
    progress_text = f"{answered}/{total} ({progress * 100:.0f}%)"

    render_top_header(st.session_state["username"], ROTULOS_RAMO[branch], progress_text)
    render_page_title(
        "Anotação concluída",
        "A sua submissão encontra-se bloqueada.",
    )

    finalized_at = st.session_state.get("finalized_at", "")
    if finalized_at:
        st.info(f"Finalização registada em: {finalized_at}.")
    st.success("Obrigado pelo seu tempo e pelo seu contributo.")

    if st.button(
        "Terminar sessão",
        use_container_width=True,
        key="btn_logout_final",
    ):
        _clear_auth_state()
        _init_session()
        st.rerun()

    _render_footer_logos()


def main() -> None:
    _init_session()

    if not st.session_state.get("authenticated", False):
        _render_login_page()
        return

    if st.session_state.get("finalized", False):
        _render_final_page()
        return

    page = st.session_state.get("page", PAGE_LOGIN)
    if page == PAGE_INSTRUCOES:
        _render_instruction_page()
    elif page == PAGE_ANOTACAO:
        _render_annotation_page()
    else:
        _render_login_page()


if __name__ == "__main__":
    main()
