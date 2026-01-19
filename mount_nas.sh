sudo mount -t nfs 172.16.13.140:/home/adminpc1/robocoin-dataset/db /mnt/db
sudo mount -t nfs -o vers=3 172.16.12.20:/volume3/docker2 /mnt/nas/synnas/docker2
sudo mount -t nfs -o vers=3 172.16.12.20:/volume1/docker /mnt/nas/synnas/docker
sudo mount -t nfs -o vers=3 172.16.12.20:/volume1/传输路径 /mnt/nas/synnas/传输路径
sudo mount -t nfs -o vers=3 172.16.12.20:/volume1/成功区 /mnt/nas/synnas/成功区
sudo mount -t nfs -o vers=3 172.16.12.20:/volume1/整理区 /mnt/nas/synnas/整理区