# Gunicorn config — extend timeout for model training on cold start
workers = 1
timeout = 120          # 2 minutes — enough for RF training + CV on free tier
worker_class = "sync"
