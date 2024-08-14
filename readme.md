Requires ffmpeg, pyaml, mkvtoolnix. Currently only compatible with unix-like operating systems (WSL should work too, but I haven't tested it yet)

Data structure involves X number of directories including the main directory of files to transcode, as well as an arbitrary number of additional directories with additional data to mux (audio transcoding also supported). All directory content must be lexicographically sortable such that they can be indexed with related file sets. That means that you should create directories that contain only the relevant data files. They don't all have to have identical names, but the names that they do have to be sortable in a relevant manner.

Todo list:

- Actually implement handling of additional data other than video
- Add progress output
- Add a resume function
- Maybe nicely tell someone that not all of their nodes are functioning
- Option to specify ffmpeg/mkvtoolnix path
- Windows support (optional path prefix for each client node)
- Simple option to blindly mux all additional data from original source file
- avx512 toggle? on my only compatible platform (7800x3d) there's a minimal if any measurable benefit to turning on avx512 in my test case (x265, 1080p 24fps anime). It's perhaps marginally faster - very low single digit %, within margin of error given other simultaneous desktop use.