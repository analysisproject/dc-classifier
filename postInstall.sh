#!/usr/bin/env bash
set -e

export PLAYWRIGHT_BROWSERS_PATH=/mount/src/dc-classifier/.playwright-browsers
python -m playwright install --with-deps --no-shell chromium
