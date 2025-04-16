#!/usr/bin/env bash

python cw_main.py $@ 2> >(grep -v '\(Pumps cannot deliver enough flow or head\|No pattern for pump\|warnings.warn\)' >&2)
