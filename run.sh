#!/usr/bin/env bash
set -euo pipefail

# Ponto de entrada oficial para avaliação do Desafio BRACIS 2026 × Jusbrasil
# Uso:
#   bash run.sh <caminho_db> <pasta_txt> <arquivo_saida>

if [ "$#" -ne 3 ]; then
    echo "Erro: número incorreto de argumentos." >&2
    echo "Uso: $0 <caminho_db> <pasta_txt> <arquivo_saida>" >&2
    exit 1
fi

DB_PATH="$1"
TEXTS_DIR="$2"
OUTPUT_PATH="$3"

if [ ! -f "$DB_PATH" ]; then
    echo "Erro: arquivo de banco de dados não encontrado: $DB_PATH" >&2
    exit 1
fi

if [ ! -d "$TEXTS_DIR" ]; then
    echo "Erro: diretório de textos não encontrado: $TEXTS_DIR" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cria diretório de destino do arquivo de saída se não existir
OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
if [ -n "$OUTPUT_DIR" ] && [ "$OUTPUT_DIR" != "." ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Cria pasta temporária para os JSONs intermediários da pipeline
TEMP_JSON_DIR="$(mktemp -d 2>/dev/null || mktemp -d -t 'bracis_json_XXXXXX')"
cleanup() {
    rm -rf "$TEMP_JSON_DIR"
}
trap cleanup EXIT

# Seleciona o interpretador Python (.venv local ou python3 do contêiner/sistema)
PYTHON_CMD="python3"
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_CMD="$SCRIPT_DIR/.venv/bin/python"
fi

echo "=== Executando Pipeline BRACIS 2026 x Jusbrasil ==="
echo "Banco de dados: $DB_PATH"
echo "Diretório de textos: $TEXTS_DIR"
echo "Arquivo de saída: $OUTPUT_PATH"
echo "Interpretador Python: $PYTHON_CMD"

"$PYTHON_CMD" "$SCRIPT_DIR/blind_pipeline.py" \
    --database "$DB_PATH" \
    --texts "$TEXTS_DIR" \
    --submission "$OUTPUT_PATH" \
    --json-dir "$TEMP_JSON_DIR"

echo "=== Execução concluída com sucesso! Saída gerada em: $OUTPUT_PATH ==="
