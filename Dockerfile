FROM python:3.10-slim

WORKDIR /app

COPY . .

# 关键！换成国内清华源，解决超时问题！
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 先安装pymilvus，避免冲突
RUN pip install --no-cache-dir pymilvus==2.3.0

# 再安装项目依赖
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "main.py", "server", "--host", "0.0.0.0", "--port", "8000"]