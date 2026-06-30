#!/bin/bash

# 0x180000008 (0x180000000 + 0x08) 에 0~37 카운트를 1초 간격으로 무한 반복 write
# 대상 필드: 5bit [4:0]

ADDR=0x180000008
MAX=37

echo "Start: count 0~${MAX} → ${ADDR} (32bit write, 1sec interval, infinite loop)"

while true; do
    for (( count=0; count<=MAX; count++ )); do
        HEX=$(printf "0x%02X" $count)
        devmem $ADDR 32 $HEX
        echo "[$(date '+%H:%M:%S')] ${ADDR} <- ${count} (${HEX})"
        sleep 1
    done
done
