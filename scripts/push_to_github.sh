#!/bin/bash
# Create GitHub repo and push story_generator_agent
set -euo pipefail

REPO_NAME="story_generator_agent"
GITHUB_USER="arunvelmurugan-jpg"
REPO_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

cd "$(dirname "$0")/.."

if ! gh auth status &>/dev/null; then
  echo "GitHub CLI not authenticated. Run:"
  echo "  gh auth login --hostname github.com --git-protocol https --web"
  exit 1
fi

# Stage and commit local prep if needed
if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
  git add .gitignore .env.example README.md render.yaml scripts/push_to_github.sh
  git commit -m "$(cat <<'EOF'
Prepare standalone GitHub repo with deployment config and API docs.

EOF
)" || true
fi

# Create repo on GitHub (skip if exists)
if gh repo view "${GITHUB_USER}/${REPO_NAME}" &>/dev/null; then
  echo "Repo already exists: ${REPO_URL}"
else
  gh repo create "${REPO_NAME}" --public --description "INVEST-compliant user story generator agent (PHTN.AI)"
fi

# Add github remote if missing
if ! git remote get-url github &>/dev/null; then
  git remote add github "${REPO_URL}"
fi

git push -u github HEAD:main --force 2>/dev/null || git push -u github HEAD:main

echo ""
echo "GitHub repo: ${REPO_URL}"
echo "Done."
