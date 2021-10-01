-- Store values for all individual song syllables

CREATE TABLE IF NOT EXISTS individual_syllable
(
    id                       INTEGER PRIMARY KEY,
    songID                   INTEGER NOT NULL,
    birdID                   STRING,
    taskName                 STRING,
    note                     STRING,
    context                  STRING,

    entropy                  REAL,
    spectroTemporalEntropy   REAL,
    entropyVar               REAL,

    FOREIGN KEY (songID) REFERENCES song (id),
    UNIQUE (songID, note)
)

