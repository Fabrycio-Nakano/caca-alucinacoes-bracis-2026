# Desafio Caça-Alucinações — BRACIS 2026 × Jusbrasil

Solução oficial para detecção, classificação e resolução canônica de citações jurídicas (reais, inventadas e incompletas) em peças processuais brasileiras.

---

## 👥 Identificação da Equipe

* **Nome da Equipe:** DnCeG
* **Integrantes:**
  * Fabrycio Leite Nakano Almada (`fabrycio@discente.ufg.br` | [@Fabrycio-Nakano](https://github.com/Fabrycio-Nakano))
  * Kauan Divino Pouso Mariano (`kauan@discente.ufg.br` | [@kauandivino](https://github.com/kauandivino))
  * Maykon Adriell Dutra (`maykonadriell@discente.ufg.br` | [@MaykonAdriell](https://github.com/MaykonAdriell))
  * Victor Emanuel da Silva Monteiro (`victor_emanuel@egresso.ufg.br` | [@victoremanuelgo](https://github.com/victoremanuelgo))

---

## 🧠 Abordagem Técnica

A solução adota uma abordagem **puramente determinística, algorítmica e de alta precisão**, dispensando modelos probabilísticos ou caixas-pretas. O pipeline é composto por quatro etapas principais:

1. **Visão Normalizada com Preservação Estrita de Offsets Unicode:**
   * Uma representação normalizada de mesmo comprimento (1:1) é gerada para permitir matching insensível a acentuação, caixa e ruídos de OCR (`[0-9olsg]`, `rn` $\leftrightarrow$ `m`), sem jamais deslocar os índices de caracteres (`start`, `end`) do texto original.
2. **Gramática Jurídica Especializada:**
   * Regras composicionais cobrem classes processuais recursivas (`AgInt nos EDcl no REsp`, `AgRg no AREsp`, `R-RP`, `AREspEl`), números com pontuação variada, dispositivos legais (`art. X, § Y, da Lei Z`, `CPC`, `CLT`) e súmulas vinculantes e ordinárias dos tribunais superiores (STF, STJ, TST, TSE).
   * Filtragem de cabeçalhos institucionais e metadados de capa para evitar falsos positivos antes do início do corpo argumentativo.
3. **Enriquecimento e Resolução Canônica Dinâmica (`canonical_metadata.py`):**
   * Ao receber qualquer banco SQLite no formato oficial, o módulo indexa e enriquece os registros *on-the-fly* em memória (identificando diplomas legais e numerações de súmulas que não constam explicitamente na coluna de texto).
   * O resolvedor opera de forma estritamente conservadora: colisões ou ambiguidades não forçam vínculos arbitrários.
4. **Classificação Tripartite Conforme o Edital:**
   * **`real`:** Citações explícitas cuja entidade existe e foi associada univocamente na base canônica;
   * **`inventada`:** Citações com identificador formal explícito (número de processo, súmula, etc.) que não foram localizadas na base canônica;
   * **`incompleta`:** Citações descritivas que indicam fonte específica (classe processual/tribunal, ano e relatoria), mas omitem número suficiente para resolução exata.

---

## ⚖️ Declaração sobre Modelos e Pesos (Offline & Zero-Downloads)

* **Uso de Pesos/Modelos:** A solução **NÃO utiliza pesos de modelos neurais, checkpoints ou embeddings externos**.
* **Execução 100% Offline:** Não há dependência de conexão com a internet, APIs externas ou downloads em tempo de execução.
* **Consumo de Recursos:** A inferência é executada em CPU leve em poucos segundos, operando com ampla folga dentro do limite estipulado de 24 GB de VRAM.

---

## 🚀 Como Executar

### Opção 1: Via Docker (Ambiente Declarado — Recomendado)

A imagem utiliza `python:3.12-slim` e já encapsula o ponto de entrada oficial.

1. **Construir a imagem:**
   ```bash
   docker build -t bracis-jusbrasil .
   ```

2. **Executar a inferência:**
   Monte o diretório contendo os dados como um volume (`-v`) e passe os caminhos relativos ao contêiner:
   ```bash
   docker run --rm \
     -v /caminho/para/dados:/dados \
     bracis-jusbrasil \
     /dados/desafio1_bracis.db /dados/txt /dados/submission.csv
   ```

---

### Opção 2: Execução Nativa (Bash)

Requisitos: **Python 3.12** com as dependências instaladas.

1. **Instalar dependências:**
   ```bash
   python3 -m pip install -r requirements.txt
   ```

2. **Executar o ponto de entrada único:**
   ```bash
   bash run.sh <caminho_db> <pasta_txt> <arquivo_saida>
   ```

   *Exemplo:*
   ```bash
   bash run.sh desafio1_bracis.db txt artifacts/submission_blind.csv
   ```

---

## 📁 Estrutura do Repositório

```text
├── Dockerfile                  # Declaração do ambiente de avaliação
├── .dockerignore               # Proteção contra inclusão de arquivos pesados na imagem
├── .gitignore                  # Exclusão de bancos (*.db), caches e arquivos temporários
├── run.sh                      # Ponto de entrada oficial único exigido pelo edital
├── blind_pipeline.py           # Pipeline determinístico de extração e classificação
├── canonical_metadata.py       # Enriquecimento e indexação canônica do SQLite
├── requirements.txt            # Dependências mínimas (numpy, pandas, pytest)
├── pytest.ini                  # Configuração da suíte de testes
├── tests/                      # 42 testes automatizados unitários e de generalização
└── README.md                   # Documentação da solução e instruções de execução
```

---

## 🧪 Testes e Validação

Para executar a suíte completa de 42 testes automatizados:
```bash
pytest -q
```
