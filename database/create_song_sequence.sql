-- By Jaerong

CREATE TABLE IF NOT EXISTS song_sequence(
    songID                   INTEGER PRIMARY KEY,
    birdID                   STRING  not null,
    taskName                 STRING  not null,
    taskSession              INT     not null,

    nbBoutsUndir             INT,
    nbBoutsDir               INT,

    transEntUndir            REAL,
    transEntDir              REAL,

    seqLinearityUndir        REAL,
    seqLinearityDir          REAL,

    seqConsistencyUndir      REAL,
    seqConsistencyDir        REAL,

    songStereotypyUndir      REAL,
    songStereotypyDir        REAL,

    UNIQUE (songID, birdID, taskName, taskSession)
);

INSERT OR IGNORE INTO song_sequence (songID, birdID, taskName, taskSession)
SELECT id, birdID, taskName, taskSession
FROM song;

