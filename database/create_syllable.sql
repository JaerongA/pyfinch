-- Information about individual syllables per sessions

CREATE TABLE IF NOT EXISTS syllable
(
    syllableID               INTEGER PRIMARY KEY,
    songID                   INTEGER NOT NULL,
    birdID                   STRING,
    taskName                 STRING,
    note                     STRING,

    nbNoteUndir              INT,
    nbNoteDir                INT,

    entropyUndir             REAL,
    entropyDir               REAL,

    spectroTemporalEntropyUndir  REAL,
    spectroTemporalEntropyDir   REAL,

    entropyVarUndir          REAL,
    entropyVarDir            REAL,

    FOREIGN KEY (songID) REFERENCES song (id),
    UNIQUE (songID, note)
)

