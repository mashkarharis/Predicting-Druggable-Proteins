TR_POS="$1"
TR_NEG="$2"
TS_POS="$3"
TS_NEG="$4"

python -u main.py "$TR_POS" "$TR_NEG" "$TS_POS" "$TS_NEG" | tee output.log

read -p "Press Enter to exit..."