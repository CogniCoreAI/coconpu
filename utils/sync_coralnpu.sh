#!/bin/bash

echo ">>> Updating dev_coralnpu from upstream..."
git checkout dev_coralnpu
git pull upstream main
git push origin dev_coralnpu

echo ">>> Rebasing main onto dev_coralnpu..."
git checkout main
git rebase dev_coralnpu

echo ">>> Done! Your 'main' is now synced with upstream and keeps your local commits on top."
echo ">>> Run 'git push origin main --force-with-lease' to update remote if needed."
