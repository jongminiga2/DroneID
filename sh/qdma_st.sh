sudo dma-ctl dev list
echo 40 | sudo tee /sys/bus/pci/devices/0005:01:00.0/qdma/qmax
echo 10 | sudo tee /sys/bus/pci/devices/0005:01:00.1/qdma/qmax
echo 10 | sudo tee /sys/bus/pci/devices/0005:01:00.2/qdma/qmax

sudo dma-ctl qdma501000 stat

sudo dma-ctl qdma501000 q add idx 0 mode st dir c2h
sudo dma-ctl qdma501000 q add idx 1 mode st dir c2h
sudo dma-ctl qdma501000 q add idx 2 mode st dir c2h
sudo dma-ctl qdma501000 q add idx 3 mode st dir c2h
sudo dma-ctl qdma501001 q add idx 0 mode mm dir h2c
sudo dma-ctl qdma501002 q add idx 0 mode mm dir c2h

sudo dma-ctl qdma501000 q start list 0 3 dir c2h
sudo dma-ctl qdma501002 q start idx 0 dir c2h
#sudo dma-ctl qdma501001 q start iqx 0 dir h2c

DEVICE="qdma501002"
# ybkim 20260305
#sudo dma-ctl qdma501000 q list 0 2
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
#sudo dma-ctl qdma501001 reg write bar 2 0x4 $PORT >/dev/null 2>&1

