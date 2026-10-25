"""
Synthetic example task: extract a structured order (JSON) from free-text
customer messages in Portuguese. Quantities are sometimes written as words
and payment is optional, so the model has to normalize, not just copy.

Usage:
    python make_dataset.py OUTPUT_PATH
"""

import sys
import json
import random

NUM_EXAMPLES = 600
SEED = 0

SYSTEM = (
    "Extraia o pedido da mensagem do cliente. Responda apenas com um JSON com as chaves "
    '"produto" (string), "quantidade" (inteiro), "cidade" (string) e "pagamento" '
    '("pix", "boleto", "cartao" ou null se não informado).'
)
PRODUCTS = ["caneca térmica", "fone bluetooth", "cadeira gamer", "mochila", "teclado mecânico",
            "garrafa de água", "luminária de mesa", "mouse sem fio", "monitor 24 polegadas", "camiseta básica",
            "tênis de corrida", "carregador portátil", "webcam", "panela de pressão", "ventilador"]
CITIES = ["São Paulo", "Recife", "Manaus", "Curitiba", "Belo Horizonte", "Porto Alegre", "Salvador",
          "Fortaleza", "Goiânia", "Belém", "Florianópolis", "Natal", "Campinas", "Vitória", "Cuiabá"]
NUMBER_WORDS = {1: "uma", 2: "duas", 3: "três", 4: "quatro", 5: "cinco", 6: "seis", 10: "dez", 12: "doze"}
PAYMENTS = {"pix": ["no pix", "via Pix", "pagando com pix"],
            "boleto": ["no boleto", "por boleto bancário"],
            "cartao": ["no cartão", "no crédito", "com cartão de crédito"]}
TEMPLATES = [
    "Oi! Quero {q} {p} pra entregar em {c}{pay}.",
    "Bom dia, gostaria de comprar {q} unidades de {p}. Endereço de entrega: {c}{pay}.",
    "preciso de {q} {p} urgente, moro em {c}{pay}",
    "Olá, vocês entregam em {c}? Se sim, manda {q} {p}{pay}.",
    "Pedido: {p}, quantidade {q}, cidade {c}{pay}. Obrigado!",
    "Me vê {q} {p}{pay}? É pra {c}.",
]


def make_example(rng):
    produto, cidade = rng.choice(PRODUCTS), rng.choice(CITIES)
    quantidade = rng.choice(list(range(1, 13)) + [15, 20, 50])
    q_text = NUMBER_WORDS[quantidade] if quantidade in NUMBER_WORDS and rng.random() < 0.4 else str(quantidade)
    pagamento = rng.choice([None, "pix", "boleto", "cartao"])
    pay_text = f", {rng.choice(PAYMENTS[pagamento])}" if pagamento else ""
    message = rng.choice(TEMPLATES).format(q=q_text, p=produto, c=cidade, pay=pay_text)
    answer = {"produto": produto, "quantidade": quantidade, "cidade": cidade, "pagamento": pagamento}
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": message},
        {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
    ], "meta": {  # ignored by training; used by benchmark.py for the per-category breakdown
        "quantity_spelled_out": q_text != str(quantidade),
        "has_payment": pagamento is not None,
    }}


if __name__ == "__main__":
    path = sys.argv[1]
    rng = random.Random(SEED)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(NUM_EXAMPLES):
            f.write(json.dumps(make_example(rng), ensure_ascii=False) + "\n")
    print(f"Wrote {NUM_EXAMPLES} examples to {path}")
