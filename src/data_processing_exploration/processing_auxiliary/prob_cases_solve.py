import re
from pathlib import Path

import pandas as pd

""" Simple functions to assist in manually solving problematic cases in the dataset. Since these cases are few, we can handle them with specific rules. 
    We go to each case url, check the decision in the text, 
    and past the initial snippet and final and the function finds the decision and updates the dataframe accordingly."""


def extrair_com_snippets(texto_completo, snippet_inicio, snippet_fim):
    """Extrai a decisão de um texto usando snippets de início e fim fornecidos manualmente.

    Args:
        texto_completo (str): Texto integral do acórdão. Se for NaN, devolve None.
        snippet_inicio (str): Excerto literal que marca o início da decisão.
        snippet_fim (str): Excerto literal que marca o fim da decisão.

    Returns:
        str or None: A decisão extraída (incluindo os snippets de início e
            fim), ou None se os snippets não forem encontrados no texto ou
            ocorrer um erro durante a extração.
    """
    if pd.isna(texto_completo):
        return None

    try:
        # Escapar caracteres especiais para regex
        inicio_escaped = re.escape(snippet_inicio)
        fim_escaped = re.escape(snippet_fim)

        # Criar regex pattern (DOTALL para incluir quebras de linha)
        pattern = f"{inicio_escaped}(.+?){fim_escaped}"

        # Procurar no texto
        match = re.search(pattern, texto_completo, re.DOTALL | re.IGNORECASE)

        if match:
            # Incluir os snippets na decisão extraída
            decisao_completa = snippet_inicio + match.group(1) + snippet_fim
            return decisao_completa.strip()
        else:
            print("❌ Snippets não encontrados no texto")
            return None

    except Exception as e:
        print(f"❌ Erro na extração: {str(e)}")
        return None


def processar_casos_manualmente_simples(df_casos_problematicos):
    """Interface interativa de linha de comandos para resolver manualmente casos problemáticos.

    Para cada caso do DataFrame, mostra a URL, a decisão conhecida e um
    preview do final do texto integral, pede ao utilizador os snippets de
    início/fim da decisão (ou 'skip' para saltar o caso), extrai a decisão
    com `extrair_com_snippets` e, após confirmação do utilizador, grava o
    resultado na coluna 'decisao_extraida_do_texto_integral' de uma cópia do
    DataFrame.

    Args:
        df_casos_problematicos (pandas.DataFrame): Casos a rever manualmente.
            Deve conter as colunas 'url', 'decisao', 'tribunal' e
            'texto_integral_completo'.

    Returns:
        pandas.DataFrame: Cópia do DataFrame de entrada com a coluna
            'decisao_extraida_do_texto_integral' preenchida para os casos
            confirmados pelo utilizador.
    """
    print("🚀 PROCESSAMENTO MANUAL DOS CASOS PROBLEMÁTICOS")
    print("=" * 60)
    print(f"📋 {len(df_casos_problematicos)} casos para processar")
    print(
        "⏱️ Estimativa: {:.0f} minutos total\n".format(
            len(df_casos_problematicos) * 2.5
        )
    )

    # Criar cópia do dataframe para modificar
    df_updated = df_casos_problematicos.copy()
    casos_processados = 0

    for idx, (df_idx, caso) in enumerate(df_casos_problematicos.iterrows()):
        print(f"\n🔍 CASO {idx + 1}/{len(df_casos_problematicos)}")
        print("=" * 40)
        print(f"📄 URL: {caso['url']}")
        print(f"⚖️ Decisão conhecida: '{caso['decisao']}'")
        print(f"🏛️ Tribunal: {caso['tribunal']}")

        # Mostrar preview do final do texto
        texto_completo = caso["texto_integral_completo"]
        if pd.notna(texto_completo) and len(texto_completo) > 500:
            print("\n📝 Preview do final do texto (últimos 500 caracteres):")
            print("-" * 50)
            print("..." + texto_completo[-500:])
            print("-" * 50)

        print("\n📋 INSTRUÇÕES:")
        print("1. Abrir URL numa nova aba")
        print("2. Ir ao final do documento")
        print(
            "3. Identificar INÍCIO da decisão (ex: 'Alcança-se, deste modo, a decisão que segue:')"
        )
        print(
            "4. Identificar FIM da decisão (ex: 'benefício do apoio judiciário concedido aos ora recorrentes).')"
        )
        print("5. Colar os snippets abaixo (ou 'skip' para pular)")

        try:
            # Input do snippet de início
            snippet_inicio = input("\n🎯 Snippet de INÍCIO da decisão: ").strip()

            if snippet_inicio.lower() == "skip":
                print("⏭️ Caso pulado")
                continue

            # Input do snippet de fim
            snippet_fim = input("🏁 Snippet de FIM da decisão: ").strip()

            if snippet_fim.lower() == "skip":
                print("⏭️ Caso pulado")
                continue

            # Extrair decisão
            decisao_extraida = extrair_com_snippets(
                texto_completo, snippet_inicio, snippet_fim
            )

            if decisao_extraida:
                print(f"\n✅ DECISÃO EXTRAÍDA ({len(decisao_extraida)} caracteres):")
                print("-" * 50)
                # Mostrar apenas os primeiros 300 caracteres para não poluir
                preview = (
                    decisao_extraida[:300] + "..."
                    if len(decisao_extraida) > 300
                    else decisao_extraida
                )
                print(preview)
                print("-" * 50)

                # Confirmar
                confirmar = (
                    input("✅ Confirmar e salvar? (s/n/retry): ").strip().lower()
                )

                if confirmar == "s":
                    # Atualizar dataframe diretamente
                    df_updated.loc[df_idx, "decisao_extraida_do_texto_integral"] = (
                        decisao_extraida
                    )
                    casos_processados += 1
                    print("✅ Caso salvo no dataframe!")

                elif confirmar == "retry":
                    print("🔄 Tentar novamente...")
                    # Processar o mesmo caso novamente
                    continue
                else:
                    print("❌ Caso rejeitado")
            else:
                print("❌ Erro na extração. Verificar snippets.")
                retry = input("🔄 Tentar novamente? (s/n): ").strip().lower()
                if retry == "s":
                    continue

        except KeyboardInterrupt:
            print("\n⏹️ Processamento interrompido pelo usuário")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")
            continue

    print("\n📊 RESUMO FINAL:")
    print(f"✅ Casos processados: {casos_processados}")
    print(f"⏭️ Casos restantes: {len(df_casos_problematicos) - casos_processados}")
    print(
        f"📈 Taxa de sucesso: {casos_processados/len(df_casos_problematicos)*100:.1f}%"
    )

    # Verificar quantos casos agora têm decisão extraída
    casos_com_decisao = df_updated["decisao_extraida_do_texto_integral"].notna().sum()
    print(f"🎯 Total de casos com decisão extraída: {casos_com_decisao}")

    return df_updated


if __name__ == "__main__":

    def main():
        """Ponto de entrada do script: lê o CSV de casos problemáticos, processa-os interativamente e grava o resultado."""
        type_case = (
            input(
                "Digite o tipo de caso ('dv' para Violência Doméstica, 'ic' para Incumprimento de Contratos): "
            )
            .strip()
            .lower()
        )
        # type_court = input("Digite o tipo de tribunal ('trp', 'trl', 'trc', 'tre', 'trg', 'stj' para Violência Doméstica; 'csm', 'jp', 'trl', 'trp', 'trc', 'tre', 'trg' para Incumprimento de Contratos): ").strip().lower()
        df_casos_problematicos = pd.read_csv(
            Path(f"./prob_cases/df_casos_problematicos_{type_case}.csv")
        )

        df_casos_fixed = processar_casos_manualmente_simples(df_casos_problematicos)

        df_casos_fixed.to_csv(
            Path(f"./fixed_cases/df_casos_problematicos{type_case}_fixed.csv"),
            index=False,
        )
        print(
            f"✅ DataFrame atualizado salvo em './fixed_cases/df_casos_problematicos_{type_case}_fixed.csv'"
        )

        return

    main()
