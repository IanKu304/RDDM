## build docker
```
docker build -t diffusion_env .
```
更新光碟映像
要更新的記得放在最後面(利用前面的快取 節省時間)

## open docker

```
docker run --gpus all -it --rm -v $(pwd):/app diffusion_env
```
這行指令的意思是：

--gpus all: 打通任督二脈，讓容器能用你的 NVIDIA 顯卡。

-it: 進入互動模式，讓你能在裡面打字下指令。

--rm: (可選) 退出容器後自動刪除這個暫時的容器實體，保持乾淨。

-v $(pwd):/app: [關鍵] 把你現在 Windows/WSL 所在的資料夾，掛載到容器裡面的 /app。這樣你在外面改程式碼，裡面會同步生效。

diffusion_env: 你剛剛建置好的映像檔名稱。

執行後，你的命令提示字元應該會變成類似 root@xxxxxxxxxxxx:/app#，代表你已經在容器裡了！