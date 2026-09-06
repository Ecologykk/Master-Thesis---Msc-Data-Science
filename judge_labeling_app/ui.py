from __future__ import annotations

from pathlib import Path

import streamlit as st

from models import Caso


def load_css(css_path: Path) -> None:
    if not css_path.exists():
        return
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_top_header(username: str, branch_label: str, progress_text: str) -> None:
    st.markdown(
        (
            "<div class='top-header'>"
            "<div><strong>Utilizador:</strong> "
            f"{username}</div>"
            "<div><strong>Ramo:</strong> "
            f"{branch_label}</div>"
            "<div><strong>Progresso:</strong> "
            f"{progress_text}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_page_title(title: str, subtitle: str = "") -> None:
    st.markdown(f"<h1 class='page-title'>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p class='page-subtitle'>{subtitle}</p>", unsafe_allow_html=True)


def render_case_block(case: Caso, show_text_fallback: bool) -> None:
    st.markdown(
        (
            "<div class='case-meta'>"
            "<div><strong>Processo:</strong> "
            f"{case.n_processo}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    if show_text_fallback:
        st.markdown("### Texto integral do caso")
        st.text_area(
            "Conteúdo do processo",
            value=case.texto_integral_completo,
            height=560,
            disabled=True,
            label_visibility="collapsed",
        )
        return

    st.components.v1.iframe(case.url, height=860, scrolling=True)


def render_choice_buttons(
    title: str,
    options: list[str],
    selected: str,
    key_prefix: str,
) -> str:
    st.markdown(f"### {title}")
    if selected:
        st.success(f"Selecionado: {selected}")
    else:
        st.info("Sem seleção.")

    new_value = selected
    for option in options:
        is_selected = option == selected
        button_type = "primary" if is_selected else "secondary"
        if st.button(
            option,
            key=f"{key_prefix}_{option}",
            use_container_width=True,
            type=button_type,
        ):
            new_value = option
    return new_value


def render_save_feedback(message: str, kind: str = "info") -> None:
    if not message:
        return
    if kind == "success":
        st.success(message)
    elif kind == "error":
        st.error(message)
    else:
        st.info(message)
