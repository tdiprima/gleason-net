#!/bin/bash

DIR="patches_prostate_seer_john_6classes"

find "$DIR" -maxdepth 1 -name '*.png' -print \
| sed -E 's/.*_([0-9]+)\.png/_\1.png/' \
| sort \
| uniq -c
