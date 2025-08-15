# image: https://docker.aityp.com/image/docker.io/python:3.12-slim?platform=linux/arm64
FROM python:3.12-slim

RUN useradd -m pyuser

WORKDIR /home/pyuser/pyiun

RUN chown -R pyuser:pyuser /usr/local/lib/python3.12/site-packages && \
    chown -R pyuser:pyuser /usr/local/bin && \
    chown -R pyuser:pyuser /home/pyuser/pyiun

USER pyuser

# 安装Python依赖
COPY --chown=pyuser:pyuser requirements.txt iun.py iun_docker.py ./
COPY --chown=pyuser:pyuser ./packages ./packages/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install ./packages/*.whl && \
    rm -f ./packages/

# 复制应用代码
COPY --chown=pyuser:pyuser ./app ./app/

# 创建必要的目录
RUN mkdir -p config logs

CMD ["python", "iun_docker.py"]
