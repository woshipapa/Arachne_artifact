git clone -b anon/dev <internal-git-repo> && (cd accelerate && curl -kLo `git rev-parse --git-dir`/hooks/commit-msg <internal-git-host> chmod +x `git rev-parse --git-dir`/hooks/commit-msg)

cd accelerate
pip install -e .