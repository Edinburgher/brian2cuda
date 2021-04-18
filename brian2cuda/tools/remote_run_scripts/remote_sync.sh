#!/bin/bash
# Synchronize current directory with remote

# DEFAULTS
# remote machine name
remote="cluster"
make_target_dir_relative_to="$HOME"

# Load configuration file
source "${BASH_SOURCE%/*}/_load_remote_config.sh" ~/.brian2cuda-remote-dev.conf

remote_home=$(ssh $remote 'echo $HOME')
local_dir=$(pwd)
relative_remote_dir=$(realpath --relative-to=$make_target_dir_relative_to $local_dir)
remote_dir=$remote_home/$relative_remote_dir

### Copy current folder over to remote
rsync -avzz \
    --rsync-path="mkdir -p $remote_dir && rsync"\
    --exclude '*.o' \
    --exclude 'tags' \
    --exclude 'results/*' \
    --exclude=".*" \
    "$local_dir"/* "$remote:$remote_dir"

echo
echo "Uploaded $local_dir to remote directory $remote_dir"
