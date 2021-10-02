-- Store values for all individual song syllables

CREATE TABLE IF NOT EXISTS individual_syllable
(
    id                       INTEGER PRIMARY KEY,
    noteIndSession           INTEGER NOT NULL, -- note index across the session
    noteIndFile              INTEGER NOT NULL, -- note index within a file
    songID                   INTEGER NOT NULL,
    fileID                   STRING,
    birdID                   STRING,
    taskName                 STRING,
    note                     STRING,
    context                  STRING,

    entropy                  REAL,
    spectroTemporalEntropy   REAL,
    entropyVar               REAL,

    FOREIGN KEY (songID) REFERENCES song (id),
    UNIQUE (noteIndFile, fileID)
)

