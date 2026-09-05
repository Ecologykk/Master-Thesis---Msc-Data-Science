"""
Teste focado para validar implementação v3++ com casos problemáticos identificados
"""

import asyncio
import json
from test11 import scrape_acordao


async def test_v3pp():
    """
    Testa 3 casos específicos:
    - Caso 1: ec03e4b00d73d1f180258481003a312b (sucesso anterior - regression test)
    - Caso 2: 3fa7a01ec60c150080258b000033c024 (V-Decisão - teste threshold adaptativo)
    - Caso 3: 6b25df26d51f64fc802584e80039b74e (3.Decisão - teste Camada 2.5)
    """

    test_urls = [
        {
            "name": "Caso 1 - Regression Test",
            "url": "https://www.dgsi.pt/jtre.nsf/134973db04f39bf2802579bf005f080b/ec03e4b00d73d1f180258481003a312b",
            "expected": "Extraction success (previous working case)",
        },
        {
            "name": "Caso 2 - Adaptive Threshold",
            "url": "https://www.dgsi.pt/jtrg.nsf/86c25a698e4e7cb7802579ec004d3832/3fa7a01ec60c150080258b000033c024",
            "expected": "V-Decisão extraction (with references at end)",
        },
        {
            "name": "Caso 3 - Camada 2.5",
            "url": "https://www.dgsi.pt/jtrg.nsf/86c25a698e4e7cb7802579ec004d3832/6b25df26d51f64fc802584e80039b74e",
            "expected": "3.DECISÃO extraction",
        },
    ]

    results = []

    for test_case in test_urls:
        print(f"\n{'='*80}")
        print(f"🧪 {test_case['name']}")
        print(f"   URL: {test_case['url']}")
        print(f"   Expectativa: {test_case['expected']}")
        print(f"{'='*80}\n")

        try:
            metadata = await scrape_acordao(test_case["url"])

            # Análise do resultado
            decisao_extraida = metadata.get("decisao_extraida_do_texto_integral")
            metadata_decisao = metadata.get("metadata_decisao", {})

            result = {
                "test_case": test_case["name"],
                "url": test_case["url"],
                "success": decisao_extraida is not None,
                "metodo_extracao": metadata_decisao.get("metodo_extracao"),
                "confianca": metadata_decisao.get("confianca"),
                "palavra_chave": metadata_decisao.get("palavra_chave_encontrada"),
                "requer_verificacao": metadata_decisao.get("requer_verificacao_manual"),
                "decisao_preview": decisao_extraida[:200] if decisao_extraida else None,
                "motivo_verificacao": metadata_decisao.get("motivo_verificacao"),
            }

            results.append(result)

            # Output detalhado
            if result["success"]:
                print(f"✅ SUCESSO!")
                print(f"   Método: {result['metodo_extracao']}")
                print(f"   Confiança: {result['confianca']}")
                print(f"   Palavra-chave: {result['palavra_chave']}")
                print(f"   Preview: {result['decisao_preview'][:100]}...")
            else:
                print(f"❌ FALHOU!")
                print(f"   Motivo: {result['motivo_verificacao']}")

        except Exception as e:
            print(f"❌ ERRO: {str(e)}")
            results.append(
                {
                    "test_case": test_case["name"],
                    "url": test_case["url"],
                    "success": False,
                    "error": str(e),
                }
            )

    # Guardar resultados
    output_file = "test_v3pp_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*80}")
    print(f"📊 RESUMO DOS TESTES")
    print(f"{'='*80}")
    print(f"Total: {len(results)} testes")
    print(f"Sucessos: {sum(1 for r in results if r.get('success'))} ✅")
    print(f"Falhas: {sum(1 for r in results if not r.get('success'))} ❌")
    print(f"\nResultados guardados em: {output_file}")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    asyncio.run(test_v3pp())
