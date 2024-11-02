This is primarily made for me, personally, to do what I want, in regards to testing encoding settings and providing multiple quality versions of anime releases. It's designed with paralellism and muxing multiple sources together in mind. That's not to say that it couldn't, for example, be used to create one single encode of one single movie preserving all metadata within - but you would have to specify all the other data you want to copy in the config file, or remux it yourself afterwards.

On the master, requires ssh, pyaml, rich, tqdm, ffmpeg on path, mkvtoolnix on path. On the slaves, requires ffmpeg on path, ssh, and lscpu if running in optimized mode. Unix-like OS only. Install WSL, install hyper-v manager, and use bridged networking for equivalent functionality on Windows machines.

Data structure involves X number of directories including the main directory of files to transcode, as well as an arbitrary number of additional directories with additional data to mux (audio transcoding also supported). All directory content must be lexicographically sortable such that they can be indexed with related file sets. That means that you should create directories that contain only the relevant data files. They don't all have to have identical names, but the names that they do have to be sortable in a relevant manner.

TODO

- Add optimizations for encoders other than x265 (despite its objective superiority in all things)
- Add un-optimized execution in general
- Add limited core use per host to save resources if desired
- ???