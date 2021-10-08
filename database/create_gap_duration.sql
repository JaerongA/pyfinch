-- Get pre-motor gap durations for all song syllables

CREATE TABLE IF NOT EXISTS gap_duration
(
    id                       INTEGER PRIMARY KEY,
    songID                   INTEGER NOT NULL,
    birdID                   STRING,
    taskName                 STRING,
    note                     STRING,
    context                  STRING,

    gap_duration             REAL,

    FOREIGN KEY (songID) REFERENCES song (id),
    UNIQUE (noteIndFile, fileID)
)

