Requires ffmpeg, pyaml, mkvtoolnix, asciimatics. Currently only compatible with unix-like operating systems (since WSL can mount nfs shares and you can use hyperv bridge networking to easily ssh in, no plans to support windows directly with the current paradigm)

Data structure involves X number of directories including the main directory of files to transcode, as well as an arbitrary number of additional directories with additional data to mux (audio transcoding also supported). All directory content must be lexicographically sortable such that they can be indexed with related file sets. That means that you should create directories that contain only the relevant data files. They don't all have to have identical names, but the names that they do have to be sortable in a relevant manner.

Todo list:

- Add progress output
- Add a resume function (implemented for individual segments, not for final file)
- Maybe nicely tell someone that not all of their nodes are functioning / other error handling/notification
- Option to specify ffmpeg/mkvtoolnix path (why?)
- Simple option to blindly mux all additional data from original source file (lowest possible priority, if that's what you want, define it in the yaml lazy)
- avx512 toggle? on my only compatible platform (7800x3d) there's a minimal if any measurable benefit to turning on avx512 in my test case (x265, 1080p 24fps anime). It's perhaps marginally faster - very low single digit %, within margin of error given other simultaneous desktop use. wake me up when someone finally test's amd's claims on zen 5 wrt video encoding... they're either straight up wrong or avx512 is more beneficial (though their claims seemed to diminish with more cores (memory bandwidth?)). in either case outlets have not sufficiently tested this... obviously avx512 is much more performance on zen 5 compared to zen 4, but the question is how much video encoding actually benefits from it...