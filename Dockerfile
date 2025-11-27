FROM python:3.11-slim

# Configurações CRÍTICAS para evitar oversubscription em Multiprocessing
# Garante que numpy/opencv/scikit usem apenas 1 thread por processo
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

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
COPY .streamlit/ .streamlit/ # Adicionado: Copia a configuração do Streamlit

# Expõe a porta do Streamlit
EXPOSE 8501

# Comando para rodar a aplicação
ENTRYPOINT ["python", "-m", "streamlit", "run", "scripts/app.py", "--server.port=8501", "--server.address=0.0.0.0"] # Removidas as flags de maxUploadSize, CORS e XSRF
