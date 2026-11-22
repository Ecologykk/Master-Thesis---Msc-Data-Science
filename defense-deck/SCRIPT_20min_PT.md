# Guião da defesa, versão Porquê → Como → O quê (português, slides em inglês)

Como usar: lê em voz alta com cronómetro, ou grava-te e transcreve. Compara o tempo real de cada slide com a coluna "Orçamento". Onde passares, estás a acrescentar detalhe que não está no slide.
Ritmo assumido: cerca de 130 palavras por minuto. O guião soma **14:15**. Ao teu ritmo medido (cerca de 75 s por passo) vais ficar acima disto; a margem até aos 20 minutos é para isso. Se no ensaio um slide passar de metade acima do orçamento, corta o segundo exemplo e fica pela frase do slide.
`[CLICK]` = próxima tecla. O deck tem 20 passos com conteúdo (o resumo e os resultados têm 2). Texto em *(itálico)* é indicação de cena, não se diz.
As etiquetas entre parênteses dizem de onde vem cada afirmação: o slide, ou a dissertação (secção, figura ou tabela do PDF).

## Orçamento

| # | Slide | Bloco | Orçamento |
|---|---|---|---|
| 0 | Capa | Porquê | 0:10 |
| 1 | Context & Motivation (DGPJ + EU) | Porquê | 0:40 |
| 2 | Context & Motivation (BES) | Porquê | 0:30 |
| 3 | Three research questions | Porquê | 0:40 |
| 4 | Three objectives | Porquê | 0:25 |
| 5 | Data and Preprocessing | Como | 0:40 |
| 6 | Method and Evaluation | Como | 0:50 |
| 7 | BERT: Sliding Window | Como | 0:35 |
| 8 | LLMs: Smart Summarization (2 passos) | Como | 1:30 |
| 9 | Headline Results | O quê | 1:10 |
| 10 | Why BERT Collapsed | O quê | 0:50 |
| 10b | The Few-Shot Paradox | O quê | 0:30 |
| 11 | SHAP on a Collapsed BERT | O quê | 0:40 |
| 12 | Reading the LLM's Reasoning | O quê | 0:45 |
| 13 | Judge Validation | O quê | 1:10 |
| 14 | RQs and Objectives | O quê | 1:00 |
| 15 | Limitations | O quê | 0:35 |
| 16 | Impact | O quê | 0:30 |
| 17 | Future Work | O quê | 0:45 |
| 19 | Thank you | | 0:05 |
| | **Total** | | **14:15** |

Blocos: Porquê 2:25 · Como 3:35 · O quê 8:00. Regra para o Porquê e o Como: no máximo 3 frases por slide; os motivos de cada escolha ficam para as perguntas.
Se faltar tempo, corta por esta ordem: o Impact (16) fundido numa frase, depois a segunda metade do zoom no resumo (8).

---

## 0. Capa (0:10)
"Boa tarde. Sou o Helton Mendonça. Esta tese é sobre prever e explicar decisões judiciais em Portugal com modelos de linguagem, orientada pelo Professor Francisco Couto e pela Juíza Sofia Wengorovius."

# PORQUÊ

## 1. Contexto: números e relatórios da UE (0:40)
*(Máximo 3 frases. Diz as fontes em voz alta; o slide tem pouco texto.)*
"A DGPJ regista 605 550 processos pendentes no fim de 2025, mais 4,1 % que em 2024. O relatório do Estado de Direito da UE via algum progresso na eficiência dos tribunais em 2025 e, em 2026, já não vê progresso. Pergunto se um modelo " *(slide 1; DGPJ Nº 103 e Nº 106; RoL 2025 p.6, 2026 p.2)*
Se perguntarem pelo valor de 2024: foi revisto entre os relatórios, por isso uso a tendência.

## 2. O caso BES (0:30)
"O melhor exemplo é o caso BES: mais de 300 crimes, mais de 20 arguidos, começou em 2014 e só chegou a tribunal em 2024. Entretanto, mais de 100 lesados morreram à espera de justiça." *(slide 2; DCIAP, RTP 15 de outubro de 2024)*
Não digas números sobre empresas: estão no apêndice.

## 3. Research questions (0:40)
"Três perguntas. Um: os LLMs batem o baseline com BERT afinado? Dois: explicam as decisões, e essas explicações batem com o raciocínio jurídico? Três: o desempenho e a explicabilidade diferem entre violência doméstica, mais sensível, e incumprimento contratual, mais documental?" *(slide 3; §1.2)*

## 4. Objectives (0:25)
"Os objetivos: o primeiro baseline para Portugal, tanto quanto sei; avaliar técnicas de deep learning e de LLMs e recomendar próximos passos; e explorar explicabilidade validada por um especialista. Volto a eles no fim." *(slide 4; §1.2)*

# COMO: escolhas e compromissos

## 5. Dados e pré-processamento (0:40)
"Dados públicos do DGSI e do CSM: 6 627 documentos, filtrados para 1 802 de incumprimento e 1 126 de violência doméstica. Os rótulos vêm de regras regex sobre o dispositivo, a decisão final de cada acórdão: são heurísticos, e mais à frente mostro como cinco juízes os verificaram." *(slide 5; Fig. 3.1)*
Não expliques os motivos de cada filtro: estão no manuscrito.

## 6. Método e avaliação (0:50)
*(Três ideias: o teste, as métricas, o bootstrap.)*
"A divisão é temporal: os casos mais antigos treinam, e os 50 mais recentes de cada domínio são o teste, para evitar fuga de informação. Comparo dois modelos: o BERT afinado e o DeepSeek, em zero-shot e few-shot." *(slide 6; §4.7)*
"Escolhi duas métricas complementares. O MCC é a principal: usa a matriz de confusão inteira e vale zero para um modelo de uma só classe, que a exatidão deixaria parecer bom. O Macro-F1 dá o mesmo peso a cada classe, para as classes raras não ficarem escondidas." *(§4.7.1)*
"Com só 50 casos de teste, um único valor esconde quanto podia variar, por isso acompanho cada métrica com um intervalo de confiança por bootstrap estratificado: faço mil reamostras do teste, mantendo a proporção das classes para as raras aparecerem sempre, calculo a métrica em cada uma, e tiro os percentis 2,5 e 97,5, que dão o intervalo a 95 %." *(§4.7.2, §4.7.4)*
"Os compromissos: só dois modelos, só 50 casos de teste por domínio, e rótulos por regex."
Só se perguntarem sobre fuga temporal: "A lógica é temporal. Houve um bug de formato de datas do DGSI no código, que deixou alguns casos posteriores ao corte no treino do BERT. O teste e os exemplos do few-shot não são afetados, e o BERT colapsou na mesma. O código é aberto e este trabalho é um baseline que se pode melhorar." (ver QA_SIMULATION.md, Q3)
Não digas: datas de corte, modelos retirados.

## 7. BERT com janelas (0:35)
"O BERT lê 512 tokens e um acórdão tem muitos mais. Por isso deslizo janelas sobrepostas, guardo 16 e faço a média dos vetores. O custo é perda de informação: a posição e a ordem do que ficou." *(slide 7; §4.3.2, §4.3.3)*

## 8. Resumo inteligente (1:30)
*(Este slide foi feito por ti, com decisões tuas e limitações. Fala com calma e com as tuas palavras.)*
"A solução que pensei foi guardar só as frases que mais importam. O problema é que os acórdãos são demasiado longos para o LLM, no hardware que eu tinha. Tentei uma janela maior, mas a memória não chegava e o modelo continuava a perder-se no texto." *(slide 8)*
"Primeiro divido o texto em frases e transformo cada uma num vetor. Depois dou uma pontuação a cada frase e guardo as de pontuação mais alta primeiro, até ao orçamento de 9 000 tokens, e ponho-as outra vez pela ordem original. No fim ficam cerca de 150 frases, uns 5 400 palavras, o que dá cerca de 8 500 tokens estimados, ou seja, quase todo o orçamento." *(slide 8; Fig. 4.1, §4.5.2)*
"As frases são as originais dos juízes: não parafraseio nada. É uma decisão e uma vantagem desta abordagem."
Se perguntarem "qual é o critério?": só a pontuação, de cima para baixo, até ao orçamento de tokens. Não falas das salvaguardas de âncoras. Só se perguntarem: "estão no manuscrito, mas não foram aplicadas no código; foi uma pequena gralha".
Atenção: o manuscrito diz "cerca de 5 400 tokens" e que nenhum caso atingiu o limite. Esses 5 400 são **palavras**. Pela estimativa do próprio código (caracteres / 4) a média é cerca de 8 500 tokens e o limite foi atingido em vários casos. O slide tem um asterisco a corrigir isto: diz "palavras" ou "tokens estimados" e, se perguntarem, que foi uma pequena gralha no manuscrito.

`[CLICK]` *(Zoom na pontuação.)*
"A pontuação mistura três coisas. A maior é a query: umas frases curtas que escrevi sobre porque é que um tribunal mantém ou altera uma decisão. Quanto mais perto delas, mais pontuação. A seguir, as âncoras: expressões típicas dos acórdãos, como 'nos termos do artigo' ou 'entende o tribunal'. E por fim o centroide, que mede quão perto a frase está da média do documento, para manter o contexto." *(slide 8 zoom; main.tex linhas 1663-1666)*
"Porquê estes pesos? Escolhi-os à mão, de forma informal, e não avaliei outros valores. Por isso, por favor, não leiam 0,52, 0,26 e 0,22 como os números ótimos. O que eu decidi foi a ordem: acredito que os fundamentos jurídicos dão mais sinal para a decisão final, e os outros dois só ajudam um pouco." *(main.tex linha 1667: "manually selected and without any formal evaluation")*
"O ideal era otimizar estes pesos com frases anotadas por juízes, ou aprendê-los com outro modelo. Deixo isso como trabalho futuro. É uma solução prática de engenharia, não uma contribuição científica."
Se perguntarem "porque estes valores?": "São pesos relativos redondos, 0,7, 0,35 e 0,3, normalizados para somar 1. Não fiz ablação."
Não digas "tentativa e erro" a não ser que te lembres de ter comparado resumos: não há registo disso no repositório.

# O QUÊ

## 9. Resultados (1:10, 2 passos)
"Cada linha é um modelo, o escuro é o MCC e o claro é o macro-F1. As barras são os intervalos de confiança, e o eixo é fixo de 0 a 1." *(slide 9; Tabela 5.11)*
*(Passo 1: violência doméstica em foco, incumprimento a cinzento.)* "Em violência doméstica, o BERT afinado fica no zero em MCC: previu sempre a mesma classe. O DeepSeek em zero-shot chega a 0,592 de MCC e 0,783 de macro-F1, o melhor resultado, assinalado com o círculo." `[CLICK]`
*(Passo 2: incumprimento em foco, violência doméstica a cinzento.)* "Em incumprimento o BERT também fica em zero. O DeepSeek zero-shot tem 0,468 de MCC e 0,663 de macro-F1, e o few-shot tem o melhor MCC, 0,523." *(main.tex 2645-2647)*
"Uma cautela: com 50 casos os intervalos são largos, o MCC em DV vai de 0,37 a 0,80. São estimativas neste protocolo, não prova de superioridade geral." *(main.tex 2671-2675)*

## 10. Porque colapsou o BERT (0:50)
"Porque é que o BERT ficou a zero? Não foi sobreajuste, foi o contrário: aprendeu pouco. Em violência doméstica previu sempre 'mantido', mesmo nos 435 casos alterados em que treinou. Em incumprimento, o MCC desce de 0,303 no treino para 0,184 na validação e 0,000 no teste." *(slide 10; §5.1.2, §5.1.3)*
"Penso que contribuíram quatro coisas: poucos dados, ajuste manual sem pesquisa, a janela deslizante com média, e regularização forte. É compatível com o que vi, mas não o testei."
"Para perceber o MCC: na violência doméstica, 0 é 'mantido' e 1 é 'alterado', a classe positiva. Em incumprimento, 0 é desfavorável, 1 parcial e 2 favorável. O BERT previu sempre a classe 0. Um modelo que prevê sempre a mesma classe fica com MCC zero." *(codificações no código, features.py)*

## 10b. O paradoxo do few-shot (0:30)
*(O professor gostou deste slide. Curto, sem explicar tudo.)*
"Esperava que dar exemplos ajudasse, mas não ajudou. Em violência doméstica, o MCC desce de 0,592 para 0,486 e o macro-F1 de 0,783 para 0,696. Em incumprimento o MCC sobe um pouco, de 0,468 para 0,523, mas o macro-F1 desce de 0,663 para 0,631." *(slide 12; §6.1.3, Tabela 5.11)*
"É compatível com um só exemplo por classe ser uma âncora fraca, mas não o testei." *(só um exemplo por classe por limite de hardware)*

## 11. SHAP (0:40)
"Para explicar o BERT usei o SHAP. Uma analogia: depois de um jogo, o SHAP diz quanto cada jogador contribuiu comparado com o jogador médio. Aqui os jogadores são as frases do acórdão, o resultado é a previsão, e o jogador médio é a resposta do modelo com o texto tapado." *(slide 11; §4.6.1)*
"Mas já vimos que este BERT previu sempre a mesma classe. É como um marcador que diz sempre 'vitória da casa', joguem quem jogarem: perguntar quem marcou não diz nada. O SHAP explica o modelo, não o mundo, e por isso não havia nada a validar com os juízes."
Só se perguntarem: num caso real as 367 frases juntas levam de 48,9 % a 83,6 % de 'mantido', mas as oito mais fortes somam só 2,65 pontos (356 frases empurram para cima e 11 para baixo). *(apêndice, exemplo SHAP DV)*

## 12. Raciocínio do LLM (0:45)
"A segunda explicação é o raciocínio do próprio LLM. Neste caso de violência doméstica o rótulo verdadeiro é 'alterado' e o DeepSeek disse 'mantido'. O texto parece jurídico e fluente: 'a sentença foi mantida', 'o recurso foi indeferido sem críticas', 'o tribunal confirmou a absolvição'. A vermelho está o que o modelo afirma. Mas o tribunal de recurso alterou a decisão." *(slide 12; apêndice, thinking log, caso 236/23.6PBHRT.L1-9)*
"Uma explicação legível não é uma explicação correta. É um problema conhecido (Turpin et al.)."

## 13. Validação por juízes (1:10)
"Construí uma aplicação de anotação e pedi a cinco juízes que verificassem os meus rótulos em 25 casos por domínio. Concordaram em 24 de 25 em violência doméstica e 21 de 25 em incumprimento. No incumprimento, são 20 de 20 se retirar um juiz cujas respostas divergiram, mas essa exclusão foi feita depois, por isso o número principal é 21." *(slide 13; §5.3, Tabela 5.15)*
"Os números são pequenos demais para estatística. Os comentários dizem mais."
"Em violência doméstica, dois juízes marcaram 'mantido', correto, mas sem confiança: um diz que a decisão errou na avaliação dos factos, e o outro concederia provimento ao recurso do Ministério Público e revogaria a decisão. Ou seja, mesmo juízes que concordam entre si contestam o acórdão em si. Em incumprimento, um juiz critica o esquema de três classes, porque ambas as partes podem recorrer e cada uma pode ganhar um recurso." *(slide 13; casos 87/25.3PGSXL-A.L1-5 e 1627/20.0T8PVZ.P1)*
"Isto diz-me que o chão de verdade é discutível, e na minha opinião rótulos assim, irrealistas para decisões reais, deviam ser abandonados em trabalho futuro."
*(Salta o vídeo da aplicação, a não ser que perguntem.)*

## 14. RQs e objetivos: respostas (1:00)
"A RQ1 está respondida: o DeepSeek zero-shot supera o BERT afinado, com MCC 0,592 e 0,468 contra 0,000, dentro deste protocolo, A RQ2 está parcialmente respondida: os dois modelos dão explicações, mas nenhuma foi validada por especialistas contra o raciocínio jurídico. A RQ3 não está respondida: os intervalos de DV e BoC sobrepõem-se, e com 50 casos por domínio não há poder estatístico." *(slide 14; §6.2, main.tex 2739-2744)*
"Quanto aos objetivos: O1 e O2 cumpridos, O3 parcial, pelo mesmo motivo da RQ2."

## 15. Limitações (0:35)
"Cinco limitações. Os rótulos são heurísticos. Os dados são pequenos: 50 casos de teste por domínio. Usei uma GPU de 6 GB, sem afinação em grande escala. Corri uma só família de LLM. E as explicações não foram validadas por especialistas jurídicos." *(slide 15; §6.5)*

## 16. Impacto (0:30)
"Contribuições: o primeiro baseline específico para Portugal; um dataset rotulado e um repositório público e reprodutível, com DOI no Zenodo; uma validação com cinco juízes através de uma aplicação feita de propósito; e uma colaboração com o poder judicial que começou com um contacto inicial." *(slide 16; Cap. 7)*
Confirma a última frase: não encontrei evidência no manuscrito. Diz-a só se a puderes sustentar.

## 17. Trabalho futuro (0:45)
"O trabalho futuro tem cinco direções, todas da dissertação. Melhorar os resumos, com seleção de frases baseada em dados e validada por especialistas. Comparar de forma sistemática mais LLMs. Pensar o uso de forma agêntica, dentro dos fluxos de trabalho das instituições. Fazer XAI bem fundamentada, validada por especialistas com uma interface simples. E fazer uma auditoria de equidade antes de qualquer utilização real." *(slide 17; §6.6)*

## 19. Obrigado (0:05)
"Obrigado. Estou disponível para perguntas."

---

## Antes de ensaiar
- **Slide 16, "colaboração com o poder judicial":** sem evidência no manuscrito nem nas notas do deck. Diz só se puderes sustentar.
- **Slide 1:** o valor de 2024 foi revisto. Diz a tendência, como no guião.
- **Contaminação:** os casos de teste de incumprimento começam a 2023-10-01. Confirma a data de corte de treino do DeepSeek, porque podem perguntar.
- **"Tentativa e erro":** só diz se te lembrares de ter comparado resumos. Não há registo no repositório.
- **Ficheiro desatualizado no repositório:** `src/modeling/llms/SUMMARIZER_ALGORITHM.md` diz pesos 0,45 / 0,40 / 0,15, o que contradiz o código e o manuscrito (0,7 / 0,35 / 0,3, normalizados para 0,52 / 0,26 / 0,22). Se alguém abrir o repositório, pode perguntar.
- **Termo:** o manuscrito diz "Fine-tuned LegalBERT" na Tabela 5.11 e o modelo é o LegalBERTimbau. Há uma nota de rodapé nos slides.

## Onde costumas alongar-te
Slides 5 (dados), 7 (BERT), 8 (resumo), 9 (resultados) e 13 (juízes). Se passares de 1:30 em algum, corta o segundo exemplo e fica pela frase do slide.
