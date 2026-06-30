#!/bin/bash

# 1. 환경 설정
DEVICE="qdma501002"
QUEUE_ID=0
ADDR=0x800009000
SIZE=9830400
COUNT=1
ADDR_RSSI=0x800A00000
SIZE_RSSI=0x30
COUNT_RSSI=1
OUTPUT_FILE="/dev/shm/received_data.bin"
#OUTPUT_FILE="/home/bstarcom/ai_leader/received_data.bin"
OUTPUT_FILE_RSSI="/home/bstarcom/rssi.txt"

# --- 수정된 부분: 인자 필수 체크 ---
# $1(첫 번째 인자)이 비어 있는지 확인 (-z)
if [ -z "$1" ]; then
    echo "Error: PORT argument is missing."
    echo "Usage: sudo $0 <PORT_NUMBER>"
    echo "Example: sudo $0 1"
    echo "i(0:pattern 1:port1 2:port2)"
    exit 1
fi

PORT=$1

# 입력값이 숫자인지 체크
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "Error: PORT must be a number (e.g., 0, 1, 2)."
    exit 1
fi
# ------------------------------

#echo "--- QDMA C2H Stream Test Start (PORT: $PORT) ---"

# 2. 기존 큐 초기화
sudo dma-ctl $DEVICE q stop idx 0 dir c2h >/dev/null 2>&1
sudo dma-ctl $DEVICE q del idx 0 dir c2h >/dev/null 2>&1

# 3. C2H 큐 설정 및 시작
#echo "[1/3] Configuring C2H Queue $QUEUE_ID..."
sudo dma-ctl $DEVICE q add idx 0 mode mm dir c2h >/dev/null 2>&1
sudo dma-ctl $DEVICE q start idx 0 dir c2h >/dev/null 2>&1

# 4. 데이터 수신 대기
#echo "[2/3] Waiting for data from FPGA (C2H)..."
#echo "Command: dma-from-device -d /dev/${DEVICE}-MM-${QUEUE_ID} -a $ADDR -s $SIZE -c $COUNT"

# port setting (전달받은 PORT 변수 사용)
sudo dma-ctl qdma501001 reg write bar 2 0x4 $PORT >/dev/null 2>&1

#echo "[3/3] Waiting for data from FPGA (C2H)..."
# reset buffer
sudo dma-ctl qdma501001 reg write bar 2 0x0 0x0 >/dev/null 2>&1
sudo dma-ctl qdma501001 reg write bar 2 0x0 0x1 >/dev/null 2>&1

# dma-from-device 실행
sudo dma-from-device -d /dev/${DEVICE}-MM-${QUEUE_ID} -a $ADDR_RSSI -s $SIZE_RSSI -c $COUNT_RSSI -f $OUTPUT_FILE_RSSI
sudo dma-from-device -d /dev/${DEVICE}-MM-${QUEUE_ID} -a $ADDR -s $SIZE -c $COUNT -f $OUTPUT_FILE

if [ $? -eq 0 ]; then
  echo "DMA Success! Data saved to $OUTPUT_FILE"
#  ls -l $OUTPUT_FILE
else
  echo "[Error] C2H Transfer Failed."
fi
#echo "--- Test Finished ---"
