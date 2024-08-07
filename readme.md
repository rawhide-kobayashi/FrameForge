Requires ffmpeg, pyaml, optionally mkvtoolnix for any complex muxing.

This project makes several assumptions about your file setup.
- Videos have a number schema in the name matching "xx" - i.e. for a series of 12 episodes, they are numbered 01-12.
- Additional files containing whatever additional data that you desire to mux also are available with the same numbering schema, but their names need not match otherwise.