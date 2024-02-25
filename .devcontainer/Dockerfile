FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

WORKDIR /src

RUN sudo apt update && sudo apt upgrade -y \
    && sudo xargs apt install -y <packages.txt; [ -f requirements.txt ] \
    && pip3 install --user -r requirements.txt; \
    pip3 install --user streamlit; \
    echo '✅ Packages installed and Requirements met'

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]