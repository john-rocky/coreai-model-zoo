#!/bin/zsh
# The device gate in one command: install the app and the staged assets (md5-checked), then run the gate.
# Build and staging come first and happen on the Mac only: ./_build.sh && ./_stage.sh
#   ./_gate.sh <udid> [extra env as JSON members]
set -u
DIR=${0:A:h}
$DIR/_install.sh "${1:?usage: _gate.sh <udid> [extra env JSON members]}" && $DIR/_run.sh "$@"
