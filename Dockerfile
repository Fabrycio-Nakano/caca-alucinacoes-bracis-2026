FROM python:3.12-slim

# Evita buffers no stdout/stderr e gravação de bytecode .pyc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Instala dependências do Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia os códigos da solução, testes e o script de execução
COPY blind_pipeline.py canonical_metadata.py run.sh pytest.ini ./
COPY tests/ ./tests/

# Concede permissão de execução ao ponto de entrada
RUN chmod +x /app/run.sh

# Ponto de entrada padrão: recebe <caminho_db> <pasta_txt> <arquivo_saida>
ENTRYPOINT ["/app/run.sh"]
