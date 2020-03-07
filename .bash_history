export PS1="root@87941ea16ed5-43c0ac7342484b10adb1772b551fcd78: "
umount -f /content/gdrive/ || umount /content/gdrive/; pkill -9 -x drive
( while `sleep 0.5`; do if [[ -d "/content/gdrive/" && "$(ls -A /content/gdrive/)" != "" ]]; then echo "google.colab.drive MOUNTED"; break; fi; done ) &
cat /tmp/tmpowhqc3ff/drive.fifo | head -1 | ( /opt/google/drive/drive --features=max_parallel_push_task_instances:10,max_operation_batch_size:15,opendir_timeout_ms:60000,virtual_folders:true --inet_family=IPV4_ONLY --preferences=trusted_root_certs_file_path:/opt/google/drive/roots.pem,mount_point_path:/content/gdrive/ --console_auth 2>&1 | grep --line-buffered -E "(Go to this URL in a browser: https://.*)$|Drive File Stream encountered a problem and has stopped"; echo "drive EXITED"; ) &
rm -rf "/root/.config/Google/DriveFS/Logs/timeouts.txt"
nohup bash -c 'tail -n +0 -F "/root/.config/Google/DriveFS/Logs/drive_fs.txt" | grep --line-buffered "QueryManager timed out" > "/root/.config/Google/DriveFS/Logs/timeouts.txt" ' < /dev/null > /dev/null 2>&1 &
disown -a
exit
