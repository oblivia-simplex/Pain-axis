"""Which registry models can this account download right now?  (reads HF_TOKEN from ~/.env or the repo .env)

    .venv-Pain-axis/bin/python local/check_access.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import models as M  # noqa: E402
from huggingface_hub import HfApi, get_token  # noqa: E402

api, tok = HfApi(), get_token()
print(f"token: {'found' if tok else 'NOT found (gated models will fail)'}\n")
for name, spec in M.REGISTRY.items():
    try:
        api.auth_check(spec.repo, repo_type="model", token=tok)
        status = "ok"
    except Exception as e:
        status = "BLOCKED: request access on the model page" if type(e).__name__ == "GatedRepoError" else f"error: {type(e).__name__}"
    print(f"{name:24s} {spec.repo:38s} {'gated ' if spec.gated else 'open  '} {status}")
