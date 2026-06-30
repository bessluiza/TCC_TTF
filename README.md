# TCC_TTF — IA generativa e desempenho operacional sob a ótica do Ajuste Tarefa-Tecnologia

Este repositório reúne o **código e os materiais da análise quantitativa** do Trabalho de
Conclusão de Curso *"Inteligência artificial generativa e desempenho operacional: uma análise
sob a ótica do ajuste tarefa-tecnologia em trabalhos intensivos em informação"*.

**Autora:** Luiza Barreto do Espírito Santo Silva
**Curso:** Engenharia de Produção — Universidade de Brasília (UnB)

---

## Sobre o projeto

O trabalho investiga em que medida o **Ajuste Tarefa-Tecnologia** (Task-Technology Fit — TTF,
de Goodhue e Thompson, 1995) explica os ganhos de produtividade da IA generativa em tarefas
intensivas em informação, lidos pelos cinco objetivos de desempenho operacional de Slack,
Chambers e Johnston (2009): qualidade, velocidade, confiabilidade, flexibilidade e custo.

Além da revisão estruturada da literatura, foi conduzida uma **análise quantitativa de dados
secundários públicos**, na qual o escore de exposição de tarefas a *Large Language Models*
(Eloundou et al., 2024) é usado como **proxy operacional do ajuste tarefa-tecnologia**, cruzado
com a base O*NET e validado contra o índice AIOE (Felten, Raj e Seamans, 2021). Este repositório
contém o código que reproduz integralmente essa análise.

---

## Estrutura do repositório

```
TCC_TTF/
├── README.md                        # este arquivo
├── analise_ttf_genai_FINAL.py       # análise no nível ocupacional: descritiva, regressão, clusters, validação
├── analise_fragmentacao_TAREFA.py   # fragmentação no nível de tarefa (Fronteira Tecnológica Fragmentada)
├── apendice_codigo.py               # código consolidado (o mesmo que consta no Apêndice A do TCC)
├── dados/                           # NÃO incluído — baixe os arquivos e coloque aqui (ver abaixo)
└── figuras/                         # gerada automaticamente ao rodar os scripts
```

---

## Dados (não incluídos no repositório)

Os arquivos de dados **não são versionados aqui** porque já possuem fonte oficial pública.
Baixe-os e coloque-os em uma pasta `dados/` na raiz do projeto:

| Arquivo | Fonte |
|---|---|
| `occ_level.csv` | Eloundou et al. (2024) — https://github.com/openai/GPTs-are-GPTs/tree/main/data |
| `full_labelset.tsv` | Eloundou et al. (2024) — https://github.com/openai/GPTs-are-GPTs/tree/main/data |
| `Task_Statements.xlsx` | O*NET Database — https://www.onetcenter.org/database.html |
| `Work_Activities.xlsx` | O*NET Database — https://www.onetcenter.org/database.html |
| `AIOE_DataAppendix.xlsx` | Felten, Raj e Seamans (2021) — disponível em https://github.com/EIG-Research/AI-unemployment |

---

## Como reproduzir

Requer **Python 3.10+**. Instale as dependências:

```bash
pip install numpy pandas matplotlib scipy scikit-learn openpyxl
```

Com os arquivos de dados na pasta `dados/`, execute:

```bash
python analise_ttf_genai_FINAL.py        # gera as figuras de exposição, regressão, clusters e validação
python analise_fragmentacao_TAREFA.py    # gera as figuras da Fronteira Tecnológica Fragmentada
```

As figuras e tabelas são salvas em `figuras/`. (Observação: o código foi escrito sem dependência
do `statsmodels` — a regressão MQO é calculada diretamente com `numpy`/`scipy`.)

---

## Principais resultados

- A exposição (proxy de TTF) é **muito maior no trabalho intensivo em informação**: mediana 0,54
  contra 0,27 das demais ocupações (Kruskal-Wallis, p < 0,001).
- Em regressão (R² = 0,73), "trabalhar com computadores" e "processar informação" **elevam** o
  ajuste; atividades físicas e, notavelmente, "analisar dados" o **reduzem** — coerente com a
  Fronteira Tecnológica Fragmentada.
- O ajuste é **fragmentado dentro de cada ocupação**: mesmo cargos de alto ajuste mantêm, em média,
  ~5% de tarefas sem ajuste. A fragmentação, contudo, não é exclusiva do trabalho intensivo em
  informação.
- **Validação convergente:** o proxy correlaciona-se fortemente com o índice AIOE de Felten, Raj
  e Seamans (correlação de Spearman ρ = 0,86; n = 683).

---

## Referências dos dados

ELOUNDOU, T.; MANNING, S.; MISHKIN, P.; ROCK, D. GPTs are GPTs: labor market impact potential of
LLMs. *Science*, v. 384, n. 6702, p. 1306-1308, 2024.

FELTEN, E.; RAJ, M.; SEAMANS, R. Occupational, industry, and geographic exposure to artificial
intelligence: a novel dataset and its potential uses. *Strategic Management Journal*, v. 42,
n. 12, p. 2195-2217, 2021.

NATIONAL CENTER FOR O*NET DEVELOPMENT. *O*NET database*. Disponível em:
https://www.onetcenter.org/database.html.

---

## Licença

Distribuído sob a licença MIT. Sinta-se livre para usar e reproduzir o código, citando o trabalho.
