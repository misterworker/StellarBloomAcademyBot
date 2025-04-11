import os

app_env = os.getenv("APP_ENV", "development")

GPT_TYPE = "gpt-4o-mini"
CORS_ORIGINS = ["https://portfolio-phi-mocha-72.vercel.app/"]

if app_env == "development":
    GPT_TYPE = "gpt-4o-mini"
    CORS_ORIGINS = ["https://portfolio-phi-mocha-72.vercel.app/", "http://localhost:3000"]
elif app_env == "prod":
    GPT_TYPE = "gpt-4o"
    CORS_ORIGINS = ["https://portfolio-phi-mocha-72.vercel.app/"]
else:
    raise ValueError(f"Unknown environment: {app_env}")
