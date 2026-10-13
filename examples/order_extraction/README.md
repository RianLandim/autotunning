# Exemplo: extração de pedidos em JSON

Tarefa sintética usada para validar o boilerplate de ponta a ponta. A partir de mensagens de clientes em português escritas de formas variadas, o modelo precisa extrair um JSON com `produto`, `quantidade`, `cidade` e `pagamento`. A quantidade às vezes vem por extenso ("duas", "doze") e o pagamento é opcional (`null`), então não basta copiar trechos do texto.

```
Entrada:  "Me vê duas mochila no pix? É pra Recife."
Saída:    {"produto": "mochila", "quantidade": 2, "cidade": "Recife", "pagamento": "pix"}
```

## Como rodar

Da raiz do repositório:

```bash
uv sync
uv run examples/run_example.py examples/order_extraction
```

O script copia `prepare.py`, `train.py` e `test.py` para `.example_runs/order_extraction/`, aplica a `config.json` nessa cópia (o `prepare.py` da raiz não é alterado) e roda:

1. `make_dataset.py`: gera 600 exemplos
2. `prepare.py`: split 480/60/60 e tokenização
3. `test.py --base`: o modelo base sem treino, no conjunto de teste
4. `train.py`: fine-tuning com LoRA por `TIME_BUDGET` segundos
5. `test.py`: o adapter treinado, no conjunto de teste

Se o fine-tuning não superar o modelo base, o script termina com erro. Os logs de cada etapa e um `results.json` ficam na pasta de trabalho.

## Configuração (`config.json`)

| Constante | Valor |
|---|---|
| `BASE_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `TIME_BUDGET` | 120 s |
| `MAX_SEQ_LEN` | 512 |
| `PRIMARY_METRIC` | `task_score` (JSON idêntico ao esperado, comparado já parseado) |

Os hiperparâmetros são os padrões do `train.py` (LoRA r=16 em todas as camadas lineares, LR 2e-4, 16 exemplos por passo).

## Resultado de referência

RTX 3070 Ti (8 GB), 13/10/2026:

| Conjunto de teste (60 exemplos) | task_score | loss |
|---|---|---|
| Modelo base, sem treino | 0,317 | 0,0990 |
| Após fine-tuning | **1,000** | 0,0000 |

Treino: 120 s, 303 passos (~10 épocas), pico de 2,8 GB de VRAM. Tempo total do script: ~4 min, sem contar o download do modelo (~1 GB).

A tarefa é fácil de propósito: o objetivo é provar que o pipeline funciona (dados → treino → avaliação → ganho mensurável), não medir o limite do modelo. Com outra GPU ou outra versão de biblioteca, os números podem variar um pouco, mas o fine-tuning deve continuar claramente acima do modelo base.

## Testando o loop autônomo

Para ver o agente trabalhando sobre o exemplo, copie a pasta de trabalho para um repositório próprio e peça ao agente uma rodada de experimentos:

```bash
cp -r .example_runs/order_extraction ~/exemplo-autotunning && cd ~/exemplo-autotunning
cp ~/caminho/para/autotunning/{program.md,pyproject.toml,uv.lock,.gitignore} . && rm -f *.log results.json
git init && git add -A && git commit -m "configuração do exemplo"
```

Depois, no agente: *"Leia o program.md e vamos começar uma nova rodada de experimentos."*
