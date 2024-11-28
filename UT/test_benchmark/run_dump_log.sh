cmd=$1
LOG_PATH=log_dump/$1.log

if [ -d log_dump/ ]; then
    sleep 1s
else
    mkdir log_dump/
fi

echo "log dump in $LOG_PATH"

bash $cmd 2>&1 | tee $LOG_PATH
