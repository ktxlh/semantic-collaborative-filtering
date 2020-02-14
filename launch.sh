#!/bin/bash
#PBS -N intelAIcloud
#PBS -e ./error_log.txt
#PBS -o ./output_log.txt

echo Starting calculation
python redo_SSim.py
echo End of calculation

