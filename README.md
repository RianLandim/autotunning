# autotunning

Boilerplate para **pesquisa autônoma de fine-tuning de qualquer modelo**, baseado no [autoresearch](https://github.com/karpathy/autoresearch) de Andrej Karpathy. Um agente de código (Claude Code, Codex etc.) configura o projeto com você e depois roda experimentos sozinho: edita o código de treino, treina por um tempo fixo, mede o resultado, mantém ou descarta a mudança e repete.

O boilerplate não vem com nenhum modelo nem dataset predefinido. Você escolhe ambos, junto com o agente, na fase de configuração.

## Como funciona

O trabalho tem duas fases, ambas descritas no `program.md`, que é a "skill" do agente:

1. **Configuração (interativa, uma vez por projeto).** O agente pergunta qual é a tarefa, verifica a GPU, propõe modelos que cabem nela, converte seus dados para o formato esperado, define a métrica com você, mede a referência do modelo sem treino e faz um teste rápido de treino. Você aprova cada escolha. No fim, a configuração é commitada e congelada.
2. **Experimentação (autônoma).** Num branch `autotunning/<tag>`, o agente só edita `train.py` e roda o loop sem parar: treina → mede → mantém ou dá `git reset`. Os resultados ficam em `results.tsv`.

## Estrutura

```
prepare.py      — configuração da tarefa, carregamento do modelo, split, tokenização, avaliação (congelado após a configuração)
train.py        — LoRA, otimizador, schedule, loop de treino (o agente edita este na experimentação)
program.md      — instruções do agente (você edita este para mudar o "método de pesquisa")
test.py         — avaliação de modelo base/adapter; o split de teste é SÓ humano
data/           — seu dataset (data/dataset.jsonl) e conversores, se houver
```

O **conjunto de teste** fica separado e o agente não pode acessá-lo. Como o agente roda centenas de experimentos contra o conjunto de validação, ele acaba se ajustando a ele. O teste, rodado por você no final, é a medida honesta.

## Começando um projeto

```bash
git clone git@github.com:RianLandim/autotunning.git meu-projeto && cd meu-projeto
git remote remove origin   # o projeto de fine-tuning vira um repo próprio
uv sync
```

Depois abra o agente nessa pasta (de preferência com as permissões liberadas, para ele rodar sozinho) e peça:

```
Leia o program.md e vamos configurar um novo projeto de fine-tuning.
```

Quando a configuração terminar:

```
Vamos começar uma nova rodada de experimentos.
```

## O que é configurado (topo do `prepare.py`)

| Constante | O que é |
|---|---|
| `BASE_MODEL` | ID do Hugging Face ou caminho local de um causal LM de texto |
| `LOAD_IN_4BIT` | QLoRA: carrega o modelo base em 4-bit para caber em GPUs menores |
| `TRUST_REMOTE_CODE` | só para modelos que exigem código customizado do Hub |
| `DATA_PATH` | dataset em jsonl no formato de chat (`{"messages": [...]}`) |
| `MAX_SEQ_LEN`, `TIME_BUDGET` | tamanho máximo dos exemplos e tempo de treino por experimento |
| `TASK_EVAL`, `PRIMARY_METRIC`, `score_prediction` | métrica: `val_loss` (genérica) ou `task_score` (gera respostas e pontua) |

Referência de VRAM para LoRA (aproximada; depende também de `MAX_SEQ_LEN` e do tamanho do vocabulário):

| VRAM  | bf16                     | 4-bit (`LOAD_IN_4BIT`) |
|-------|--------------------------|------------------------|
| 8 GB  | até ~1.5B                | até ~4B                |
| 16 GB | até ~3–4B                | até ~8B                |
| 24 GB | até ~7–8B (com gradient checkpointing) | até ~14B |
| 80 GB | até ~14B+                | até ~70B               |

Modelos sem chat template (modelos base, não-instruct) usam um formato simples de fallback (`### User:` / `### Assistant:`).

## Usando o resultado

Cada execução de `train.py` salva o adapter LoRA em `out/adapter`. Para reproduzir o melhor experimento, faça checkout do último commit mantido do branch `autotunning/<tag>` e rode `uv run train.py`. Depois:

```bash
uv run test.py --base     # modelo base no teste (referência)
uv run test.py            # adapter no teste
```

Para servir o modelo, carregue o base com `PeftModel.from_pretrained(model, "out/adapter")` ou faça merge com `model.merge_and_unload()` e salve o modelo completo.

## Limitações

- Uma GPU NVIDIA, um processo. Sem treino distribuído.
- Só modelos de texto (`AutoModelForCausalLM`). Modelos multimodais ou de classificação (encoders) exigem adaptar `prepare.py` e `train.py`.
- Modelos de raciocínio cujo chat template reescreve o histórico (ex.: remove `<think>`) não funcionam sem adaptar `apply_template`.

## Licença

MIT. Baseado no [autoresearch](https://github.com/karpathy/autoresearch) (MIT, Andrej Karpathy); veja `LICENSE`.
