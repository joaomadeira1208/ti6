FROM python:3.11-slim

# Instala dependências do sistema necessárias para o OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia os arquivos de requisitos
COPY requirements.txt .

# Instala dependências Python
# Removemos versões específicas que podem conflitar com Linux se foram geradas no Mac,
# mas idealmente usamos o requirements.txt. Vamos tentar instalar direto.
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código fonte e modelos
COPY src/ src/
COPY scripts/ scripts/
COPY models/ models/

# Expõe a porta do Streamlit
EXPOSE 8501

# Comando para rodar a aplicação
ENTRYPOINT ["python", "-m", "streamlit", "run", "scripts/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
