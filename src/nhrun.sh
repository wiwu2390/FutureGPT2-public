NAME=nohup_$(date +%Y%m%d-%H%M%S)
nohup "$@" 1> $NAME.out 2> $NAME.err & echo $! > $NAME.pid

