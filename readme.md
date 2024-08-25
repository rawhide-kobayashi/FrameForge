On the master, requires ssh, pyaml, rich, tqdm, ffmpeg on path, mkvtoolnix on path. On the slaves, requires ffmpeg on path, ssh. Currently only compatible with unix-like operating systems (since WSL can mount nfs shares and you can use hyperv bridge networking to easily ssh in, no plans to support windows directly with the current paradigm). Make sure that the user that runs the file on the master can ssh into each slave passwordlessly and also into itself (@localhost).

Data structure involves X number of directories including the main directory of files to transcode, as well as an arbitrary number of additional directories with additional data to mux (audio transcoding also supported). All directory content must be lexicographically sortable such that they can be indexed with related file sets. That means that you should create directories that contain only the relevant data files. They don't all have to have identical names, but the names that they do have to be sortable in a relevant manner.