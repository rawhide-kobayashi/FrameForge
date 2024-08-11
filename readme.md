Requires ffmpeg, pyaml, optionally mkvtoolnix for any complex muxing.

This project makes several assumptions about your file setup.
- Videos have a number schema in the name matching "xx" - i.e. for a series of 12 episodes, they are numbered 01-12.
- Additional files containing whatever additional data that you desire to mux also are available with the same numbering schema, but their names need not match otherwise.

Todo list:

- Actually implement handling of additional data other than video
- Add progress output
- Add a resume function
- Maybe nicely tell someone that not all of their nodes are functioning
- Option to specify ffmpeg/mkvtoolnix path
- Windows support (optional path prefix for each client node)
- Simple option to blindly mux all additional data from original source file