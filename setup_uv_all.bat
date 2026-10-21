set GIT_CONFIG_COUNT=1
set GIT_CONFIG_KEY_0=url.https://%KENNC_REPO_PAT%@github.com/.insteadOf
set GIT_CONFIG_VALUE_0=https://github.com/
uv lock --upgrade
uv sync --extra all
call .venv\Scripts\activate.bat
